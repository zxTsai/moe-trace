"""CPU-only Ascend adapter tests; no import of vLLM, TorchNPU or CANN."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from moe_trace.adapters import ascend_vllm_0_23_0 as adapter
from moe_trace.config import TraceConfig
from moe_trace.errors import ConfigError, IntegrityError
from moe_trace.reader import TraceReader
from moe_trace.session import TraceSession
from moe_trace.validation.ascend_cpu import owner, runner_base, run_validation


@pytest.fixture(autouse=True)
def reset():
    adapter.reset_for_tests()
    yield
    assert adapter._active.get() is None
    adapter.reset_for_tests()


@pytest.fixture
def active(tmp_path, monkeypatch):
    config = TraceConfig(enabled=True, output_root=tmp_path, save_token_ids=True,
                         save_expert_weights=True, shard_records=2, buffer_records=4)
    session = TraceSession(config, model={"architecture": "GraniteMoeForCausalLM"},
                           moe_layers=[0, 1], experts_per_layer={0: 4, 1: 4}, top_k=2)
    monkeypatch.setattr(adapter, "_config", config)
    monkeypatch.setattr(adapter, "_session", session)
    return session


def batch():
    runner = runner_base()
    runner.input_batch = NS(req_ids=["decode", "prefill"], num_computed_tokens_cpu=np.array([2, 0]))
    runner.requests = {"decode": NS(num_prompt_tokens=2, get_token_id=lambda pos: [1, 2, 3][pos]),
                       "prefill": NS(num_prompt_tokens=2, get_token_id=lambda pos: [10, 11][pos])}
    scheduler = NS(num_scheduled_tokens={"prefill": 2, "decode": 1}, total_num_scheduled_tokens=3,
                   scheduled_cached_reqs=NS(resumed_req_ids=set()), preempted_req_ids=set())
    return runner, scheduler, torch.tensor([3, 10, 11]), torch.tensor([2, 0, 1])


def routes(layer=0, ids=None, weights=None, **kwargs):
    adapter.record_routes(f"model.layers.{layer}.experts",
                          torch.tensor([[0, 1], [1, 2], [2, 3]]) if ids is None else ids,
                          torch.tensor([[.75, .25]] * 3) if weights is None else weights, **kwargs)


def test_mapping_uses_reordered_batch_not_scheduler_dict_order():
    rows = adapter.build_rows(*batch())
    assert [(r.request_id, r.token_position, r.phase, r.token_id) for r in rows] == [
        ("decode", 2, "decode", 3), ("prefill", 0, "prefill", 10),
        ("prefill", 1, "prefill", 11)]


@pytest.mark.parametrize("kind", ["padding", "missing", "float_ids", "float_positions", "positions",
                                  "tokens", "unknown_request", "zero", "negative", "float_count",
                                  "bool_count", "total", "duplicate", "counter", "negative_counter",
                                  "embeddings", "2d", "negative_token", "empty"])
def test_invalid_mapping_rejected(kind):
    runner, scheduler, ids, positions = batch()
    if kind == "padding": ids = torch.tensor([3, 10, 11, 0])
    if kind == "missing": ids = ids[:2]
    if kind == "float_ids": ids = ids.float()
    if kind == "float_positions": positions = positions.float()
    if kind == "positions": positions = torch.tensor([2, 1, 0])
    if kind == "tokens": ids = torch.tensor([4, 10, 11])
    if kind == "unknown_request": scheduler.num_scheduled_tokens["extra"] = 1
    if kind in {"zero", "negative", "float_count", "bool_count"}:
        scheduler.num_scheduled_tokens["decode"] = {"zero": 0, "negative": -1,
                                                   "float_count": 1.0, "bool_count": True}[kind]
    if kind == "total": scheduler.total_num_scheduled_tokens = 4
    if kind == "duplicate": runner.input_batch.req_ids = ["decode", "decode"]
    if kind == "counter": runner.input_batch.num_computed_tokens_cpu = []
    if kind == "negative_counter": runner.input_batch.num_computed_tokens_cpu[0] = -1
    if kind == "embeddings": ids = None
    if kind == "2d": positions = positions.reshape(1, -1)
    if kind == "negative_token": ids[0] = -1
    if kind == "empty": runner.input_batch.req_ids = []
    with pytest.raises(IntegrityError):
        adapter.build_rows(runner, scheduler, ids, positions)


@pytest.mark.parametrize("versions", [("0.29.0", "0.23.0"), ("0.23.0", "0.22.0"),
                                      ("0.23.0rc1", "0.23.0"), ("0.23.0", "0.23.0.dev1")])
def test_version_mismatch(versions):
    with pytest.raises(ConfigError): adapter.check_versions(*versions)


@pytest.mark.parametrize("section,attr,value", [
    ("model_config", "enforce_eager", False), ("model_config", "quantization", "int8"),
    ("model_config", "enable_return_routed_experts", True), ("model_config", "enable_prompt_embeds", True),
    ("cache_config", "enable_prefix_caching", True), ("scheduler_config", "async_scheduling", True),
    ("parallel_config", "distributed_executor_backend", "mp"),
    *[("parallel_config", attr, 2) for attr in ["tensor_parallel_size", "pipeline_parallel_size",
                                              "data_parallel_size", "decode_context_parallel_size",
                                              "prefill_context_parallel_size"]],
    ("parallel_config", "enable_expert_parallel", True), ("parallel_config", "use_ubatching", True),
    ("parallel_config", "use_sequence_parallel_moe", True),
])
def test_unsupported_config(section, attr, value):
    subject = owner()
    setattr(getattr(subject.vllm_config, section), attr, value)
    with pytest.raises(ConfigError): adapter._reject_unsupported(subject, TraceConfig(enabled=True))


@pytest.mark.parametrize("attr", ["speculative_config", "kv_transfer_config", "ec_transfer_config", "lora_config"])
def test_unsupported_optional_config(attr):
    subject = owner()
    setattr(subject.vllm_config, attr, object())
    with pytest.raises(ConfigError): adapter._reject_unsupported(subject, TraceConfig(enabled=True))


def test_other_architecture_rejected():
    subject = owner()
    subject.vllm_config.model_config.hf_config.architectures = ["Qwen3MoeForCausalLM"]
    with pytest.raises(ConfigError): adapter._reject_unsupported(subject, TraceConfig(enabled=True))


def test_v2_runner_rejected():
    subject = owner()
    subject.vllm_config.use_v2_model_runner = True
    with pytest.raises(ConfigError): adapter._reject_unsupported(subject, TraceConfig(enabled=True))


def test_terminal_reason_cannot_be_fabricated(active):
    req = NS(request_id="R", resumable=False, get_finished_reason=lambda: None)
    adapter.request_started(owner(), req)
    with pytest.raises(IntegrityError, match="finished reason"):
        adapter.request_finished(owner(), req)
    assert TraceReader(active.run_dir).status["state"] == "failed"


@pytest.mark.parametrize("attr", ["enable_flashcomm1", "enable_flashcomm2_parallel_size", "enable_fused_mc2",
                                 "multistream_overlap_gate", "multistream_overlap_shared_expert",
                                 "enable_balance_scheduling"])
def test_unsupported_ascend_runner(attr):
    subject = runner_base()
    setattr(subject.ascend_config, attr, True)
    with pytest.raises(ConfigError): adapter._check_runner(subject)


@pytest.mark.parametrize("kind", ["graph", "eplb", "profiling", "enpu", "async"])
def test_unsupported_execution_mode(kind):
    subject = runner_base()
    if kind == "graph": subject.compilation_config.cudagraph_mode.name = "FULL"
    if kind == "eplb": subject.ascend_config.eplb_config.dynamic_eplb = True
    if kind == "profiling": subject.ascend_config.profiling_chunk_config.enabled = True
    if kind == "enpu": subject.enable_enpu = True
    if kind == "async": subject.use_async_scheduling = True
    with pytest.raises(ConfigError): adapter._check_runner(subject)


def test_session_initialization_checks_versions_without_device_imports(tmp_path, monkeypatch):
    subject = owner()
    subject.vllm_config.model_config.model = "fixture"
    hf = subject.vllm_config.model_config.hf_config
    hf.num_hidden_layers, hf.num_local_experts, hf.num_experts_per_tok = 2, 4, 2
    monkeypatch.setattr(adapter.importlib.metadata, "version", lambda name: "0.23.0")
    session = adapter._get_session(subject, TraceConfig(enabled=True, output_root=tmp_path))
    assert TraceReader(session.run_dir).metadata["framework"]["backend"] == "ascend"


def test_disabled_no_dependency_or_tensor_access(tmp_path, monkeypatch):
    path = tmp_path / "disabled.yaml"
    path.write_text("trace:\n  enabled: false\n")
    monkeypatch.setenv("MOE_TRACE_CONFIG", str(path))
    adapter.request_started(None, None)
    adapter.request_finished(None, None)
    with adapter.trace_execution(None, None, None, None):
        adapter.record_routes(None, None, None)
    adapter.close_trace()


@pytest.mark.parametrize("kwargs", [{"zero_experts": 1}, {"force_load_balance": True},
                                    {"dynamic_eplb": True}, {"comm_type": "ALLTOALL"}])
def test_bad_route_backend_fails_run(active, kwargs):
    with pytest.raises(ConfigError):
        with adapter.trace_execution(*batch()): routes(**kwargs)
    reader = TraceReader(active.run_dir)
    assert reader.status["state"] == "failed"
    assert not reader.complete


@pytest.mark.parametrize("kind", ["missing_layer", "duplicate_layer", "float_ids", "out_of_range",
                                 "bad_shape", "nan_weights", "inf_weights", "unknown_layer",
                                 "unknown_name", "weights_shape", "forward_failure", "overflow", "negative_weights"])
def test_corrupt_routes_cannot_complete(active, kind):
    with pytest.raises((IntegrityError, RuntimeError)):
        with adapter.trace_execution(*batch()):
            if kind == "float_ids": routes(ids=torch.zeros(3, 2))
            elif kind == "out_of_range": routes(ids=torch.full((3, 2), 4))
            elif kind == "bad_shape": routes(ids=torch.zeros(2, 2, dtype=torch.int64))
            elif kind == "nan_weights": routes(weights=torch.full((3, 2), float("nan")))
            elif kind == "inf_weights": routes(weights=torch.full((3, 2), float("inf")))
            elif kind == "unknown_layer": routes(layer=9)
            elif kind == "unknown_name": adapter.record_routes("not-a-layer", None, None)
            elif kind == "weights_shape": routes(weights=torch.zeros(2, 2))
            elif kind == "overflow": routes(ids=torch.full((3, 2), 2 ** 32))
            elif kind == "negative_weights": routes(weights=torch.full((3, 2), -1.0))
            elif kind == "forward_failure": raise RuntimeError("consumer failed")
            else: routes()
            if kind == "duplicate_layer": routes()
            if kind != "missing_layer": routes(layer=1)
    assert TraceReader(active.run_dir).status["state"] == "failed"


def test_copies_consumed_tensor_values_not_later_mutation(active):
    ids = torch.tensor([[0, 1], [1, 2], [2, 3]])
    weights = torch.tensor([[.75, .25]] * 3, dtype=torch.bfloat16)
    with adapter.trace_execution(*batch()):
        routes(ids=ids, weights=weights)
        ids.fill_(0)
        weights.fill_(0)
        routes(layer=1)
    active.request_event("decode", "finished", reason="stop")
    active.request_event("prefill", "finished", reason="stop")
    adapter.close_trace()
    rows = [r for r in TraceReader(active.run_dir).detail_rows() if r["layer_id"] == 0]
    assert [r["expert_ids"] for r in rows] == [[0, 1], [1, 2], [2, 3]]
    assert all(r["expert_weights"] == [.75, .25] for r in rows)


@pytest.mark.parametrize("kind", ["preempted", "resumed", "nested"])
def test_recomputation_and_nested_execution_rejected(active, kind):
    runner, sched, ids, positions = batch()
    if kind == "preempted": sched.preempted_req_ids = {"decode"}
    if kind == "resumed": sched.scheduled_cached_reqs.resumed_req_ids = {"decode"}
    with pytest.raises((ConfigError, IntegrityError)):
        with adapter.trace_execution(runner, sched, ids, positions):
            if kind == "nested":
                with adapter.trace_execution(runner, sched, ids, positions): pass
    assert not TraceReader(active.run_dir).complete


def test_shutdown_requires_terminal_requests(active):
    with adapter.trace_execution(*batch()):
        routes()
        routes(layer=1)
    with pytest.raises(IntegrityError): adapter.close_trace()
    assert not TraceReader(active.run_dir).complete


@pytest.mark.parametrize("save_ids,save_weights", [(False, False), (True, False), (False, True), (True, True)])
def test_end_to_end_optional_fields(tmp_path, save_ids, save_weights):
    config = TraceConfig(enabled=True, output_root=tmp_path / "traces", shard_records=4,
                         buffer_records=8, save_token_ids=save_ids, save_expert_weights=save_weights)
    report = run_validation(tmp_path, config=config)
    assert report["route_records"] == 30
    assert report["mixed_phase_executions"] == 2
    assert report["multi_request_executions"] == 4
    assert not report["npu_verified"]
    data = json.loads((tmp_path / "details.json").read_text())
    assert all(("token_id" in r) == save_ids and ("expert_weights" in r) == save_weights
               for r in data["records"])


@pytest.mark.parametrize("phases,layers,expected", [(("decode",), None, 12),
                                                    (("prefill",), None, 18),
                                                    (("prefill", "decode"), (1,), 15)])
def test_filtered_validation(tmp_path, phases, layers, expected):
    config = TraceConfig(enabled=True, output_root=tmp_path / "traces", phases=phases, layers=layers,
                         save_token_ids=True, save_expert_weights=True)
    report = run_validation(tmp_path, config=config)
    assert report["route_records"] == expected


def test_json_export_overwrites_previous_run(tmp_path):
    first = run_validation(tmp_path)
    second = run_validation(tmp_path)
    data = json.loads((tmp_path / "details.json").read_text())
    assert len(data["records"]) == 30
    assert data["run_id"] != Path(first["run_dir"]).name
    assert data["run_id"] == Path(second["run_dir"]).name


def test_independent_consumer_detects_wrong_request_mapping(tmp_path, monkeypatch):
    original = adapter.build_rows
    monkeypatch.setattr(adapter, "build_rows", lambda *args: list(reversed(original(*args))))
    with pytest.raises(AssertionError): run_validation(tmp_path)
    summary = json.loads((tmp_path / "cpu-validation.json").read_text())
    assert summary["state"] == "failed"


def test_failed_revalidation_invalidates_old_success(tmp_path, monkeypatch):
    run_validation(tmp_path)
    original = adapter.record_routes
    def wrong_order(name, ids, weights, **kwargs):
        if ids is None:
            return original(name, ids, weights, **kwargs)
        return original(name, ids.flip(-1), weights.flip(-1), **kwargs)
    monkeypatch.setattr(adapter, "record_routes", wrong_order)
    with pytest.raises(AssertionError, match="expert_ids mismatch"):
        run_validation(tmp_path)
    assert json.loads((tmp_path / "cpu-validation.json").read_text())["state"] == "failed"


def test_request_labels_survive_worker_cached_request(active, monkeypatch):
    config = TraceConfig(enabled=True, output_root=active.config.output_root,
                         request_labels=("selected",), save_token_ids=True, save_expert_weights=True)
    active.config = config
    monkeypatch.setattr(adapter, "_config", config)
    request = NS(request_id="decode", moe_trace_label="selected", resumable=False)
    adapter.request_started(owner(), request)
    with adapter.trace_execution(*batch()):
        routes()
        routes(layer=1)
    active.request_event("decode", "finished", reason="stop")
    adapter.close_trace()
    reader = TraceReader(active.run_dir)
    assert {r["request_id"] for r in reader.detail_rows()} == {"decode"}
