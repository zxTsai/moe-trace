from dataclasses import replace
from types import SimpleNamespace as NS
import json

import pytest

from moe_trace.adapters import ascend_v41_debug as adapter
from moe_trace.distributed.debug_layout import selector_rows
from moe_trace.distributed.decode import Worker
from moe_trace.distributed.storage import collect_run
from moe_trace.errors import IntegrityError, ConfigError

torch = pytest.importorskip("torch")


def runtime():
    parallel = NS(tensor_parallel_size=2, data_parallel_size=8, pipeline_parallel_size=1,
                  enable_expert_parallel=True)
    cfg = NS(model_config=NS(enforce_eager=True, quantization="ascend",
                            hf_config=NS(architectures=["DeepseekV41ForCausalLM"])),
             parallel_config=parallel, scheduler_config=NS(async_scheduling=False), speculative_config=None,
             kv_transfer_config=NS(kv_role="kv_consumer", kv_connector="MooncakeHybridConnector"),
             cache_config=NS(enable_prefix_caching=False))
    return NS(vllm_config=cfg, ascend_config=NS(eplb_config=NS(dynamic_eplb=False)))


def inputs():
    # Actual batch B,A, but schedule dictionary A,B; A has both prompt and decode rows.
    requests = {"A": NS(num_prompt_tokens=2, get_token_id=lambda pos: [10, 11, 12][pos]),
                "B": NS(num_prompt_tokens=1, get_token_id=lambda pos: [20, 21, 22][pos])}
    runner = runtime()
    runner.input_batch = NS(req_ids=["B", "A"], num_computed_tokens_cpu=[1, 1])
    runner.requests = requests
    scheduler = NS(num_scheduled_tokens={"A": 2, "B": 1}, total_num_scheduled_tokens=3)
    return runner, scheduler, torch.tensor([21, 11, 12, 0]), torch.tensor([1, 1, 2, 0])


def test_real_input_order_positions_and_decode_filter():
    runner, scheduler, ids, pos = inputs()
    rows = adapter.build_inputs(runner, scheduler, ids, pos, 1, "engine")
    assert [(i, r.local_request, r.token_id, r.position) for i, r in rows.items()] == [
        (0, "B", 21, 1), (2, "A", 12, 2)]
    ids[2] = 13
    with pytest.raises(IntegrityError, match="mismatch"):
        adapter.build_inputs(runner, scheduler, ids, pos, 1, "engine")


@pytest.mark.parametrize("kind,padding,sp,n,rank,expected", [
    ("PrepareAndFinalizeWithAll2All", None, False, 2, 0, [(0, 0)]),
    ("PrepareAndFinalizeWithAll2All", None, False, 2, 1, [(2, 0)]),
    ("PrepareAndFinalizeWithMC2", 8, False, 4, 0, [(0, 0), (2, 2)]),
    ("PrepareAndFinalizeWithMC2", 8, False, 4, 1, []),
    ("PrepareAndFinalizeWithAll2All", None, True, 2, 1, [(2, 0)]),
    ("PrepareAndFinalizeWithMC2", 8, True, 4, 1, [(2, 0)]),
])
def test_tp_mapping(kind, padding, sp, n, rank, expected):
    assert selector_rows(kind, input_size=4, valid_indices=[0, 2], tp_size=2, tp_rank=rank,
                         dp_rank=0, output_size=n, sequence_parallel=sp, padded_size=padding) == expected


def test_explicit_dp_and_ep_gather_layouts():
    common = dict(input_size=4, valid_indices=[0, 2], tp_size=2, tp_rank=0, dp_rank=1)
    assert selector_rows("PrepareAndFinalizeWithAllGather", **common, output_size=12,
                         dp_sizes=[2, 4, 1]) == [(0, 4), (2, 6)]
    assert selector_rows("PrepareAndFinalizeWithAllGather", **common, output_size=8,
                         sequence_parallel=True, dp_sizes=[2, 4, 1], ep_sizes=[1, 1, 2, 2, 1, 1]) == [(0, 2), (2, 4)]
    assert selector_rows("PrepareAndFinalizeWithAllGather", **{**common, "tp_rank": 1},
                         output_size=7, sequence_parallel=True, dp_sizes=[2, 4, 1]) == []
    with pytest.raises(IntegrityError, match="layout"):
        selector_rows("PrepareAndFinalizeWithAllGather", **common, output_size=7, dp_sizes=[2, 4, 1])


def test_unknown_prepare_is_not_guessed_from_shape():
    with pytest.raises(ConfigError, match="unsupported"):
        selector_rows("unknown", input_size=2, valid_indices=[0], tp_size=2, tp_rank=0,
                      dp_rank=0, output_size=2)


def test_debug_profile_guard():
    runner = runtime()
    adapter.check_runtime(runner, NS(name="NONE"))
    runner.vllm_config.speculative_config = NS(method="dspark")
    with pytest.raises(ConfigError, match="no DSpark"):
        adapter.check_runtime(runner, NS(name="NONE"))


def test_runtime_capture_and_storage_without_user_supplied_bindings(tmp_path, monkeypatch):
    runner, scheduler, ids, pos = inputs()
    layer = NS(layer_name="model.layers.20.mlp.experts", router=object(),
               moe_config=NS(is_sequence_parallel=False))
    runner.model = NS(named_modules=lambda: [("layer", layer)])
    cfg = {"enabled": True, "output_root": str(tmp_path), "run_id": "debug",
           "node_id": "D0", "save_token_ids": False, "save_expert_weights": True}
    worker = Worker("ascend-v41-debug", "debug", "D0", "dp-0-tp-0", 0, 0, 0, 0)
    session = adapter.DebugSession(runner, cfg, worker)
    monkeypatch.setattr(adapter, "read_config", lambda: cfg)
    monkeypatch.setattr(adapter, "_get_session", lambda *args: session)
    prepare = type("PrepareAndFinalizeWithAll2All", (), {})()
    with adapter.trace_target(runner, scheduler, ids, pos, NS(name="NONE")):
        adapter.capture_routes(layer, torch.tensor([[3, 1], [1, 2]]), torch.ones(2, 2),
                               prepare, NS(), NS(), 4)
        adapter.capture_consumer(layer, torch.tensor([[2, 3, 4], [3, 0, 4]]), torch.ones(2, 3))
    session.writer.close()
    _, joined = collect_run(session.directory / "trace")
    assert len(joined) == 1  # TP0 owns rows 0,1; row1 is prompt and excluded.
    route = joined[0][0]
    assert route.row.local_request == "B" and route.expert_ids == (3, 1)
    assert route.row.token_id is None and route.row.input_role == "redacted"
    consumer = json.loads(session.consumer_path.read_text())
    assert consumer["expert_ids"] == [[2, 3, 4]]


def test_disabled_context_and_warmup_do_not_capture(monkeypatch):
    monkeypatch.delenv("MOE_TRACE_V41_DEBUG_CONFIG", raising=False)
    with adapter.trace_target(None, None, None, None, None):
        adapter.capture_routes(None, None, None, None, None, None, None)
        adapter.capture_consumer(None, None, None)


def test_redacted_route_token_contract():
    runner, scheduler, ids, pos = inputs()
    row = list(adapter.build_inputs(runner, scheduler, ids, pos, 1, "engine").values())[0]
    assert replace(row, token_id=None, input_role="redacted").token_id is None
    with pytest.raises(IntegrityError):
        replace(row, input_role="redacted")
