"""DeepSeek V4 D-side runtime-K MTP capture; A3 acceptance pending."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from dataclasses import replace
import json
import hashlib
import os
from pathlib import Path
import re

from ..distributed.debug_layout import selector_rows
from ..distributed.decode import Worker, Route, DecodeRow, CandidateResult
from ..distributed.storage import create_run, WorkerWriter
from ..errors import ConfigError, IntegrityError
from ..io import atomic_json

PROFILE = "deepseek-v4-mtp-debug"
_active = ContextVar("deepseek_v4_mtp_capture", default=None)
_session = None


def read_config():
    path = os.environ.get("MOE_TRACE_V4_MTP_CONFIG")
    if not path:
        return None
    cfg = json.loads(Path(path).read_text())
    if set(cfg) != {"enabled", "output_root", "run_id", "node_id", "save_token_ids", "save_expert_weights"}:
        raise ConfigError("invalid V4 capture keys")
    if any(type(cfg[k]) is not bool for k in ("enabled", "save_token_ids", "save_expert_weights")):
        raise ConfigError("capture flags must be bool")
    if not cfg["enabled"]:
        return None
    for key in ("run_id", "node_id"):
        if not isinstance(cfg[key], str) or cfg[key] in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", cfg[key]):
            raise ConfigError("unsafe capture identity")
    if not isinstance(cfg["output_root"], str) or not Path(cfg["output_root"]).is_absolute():
        raise ConfigError("output_root must be absolute")
    if any(os.environ.get(k) for k in ("MOE_TRACE_CONFIG", "MOE_TRACE_V41_DEBUG_CONFIG", "MOE_TRACE_V41_SOURCE_HOOKS", "MOE_TRACE_QWEN36_CONFIG", "MOE_TRACE_V4_DEBUG_CONFIG")):
        raise ConfigError("do not combine capture adapters")
    return cfg


def check_runtime(runner, mode):
    cfg = runner.vllm_config
    if (getattr(mode, "name", None) != "NONE" or not cfg.model_config.enforce_eager
            or cfg.scheduler_config.async_scheduling
            or getattr(cfg, "use_v2_model_runner", False)):
        raise ConfigError("MTP capture requires V1 eager synchronous")
    spec = cfg.speculative_config
    if (spec is None or spec.method != "mtp" or type(spec.num_speculative_tokens) is not int
            or spec.num_speculative_tokens < 1
            or getattr(spec, "disable_padded_drafter_batch", False)):
        raise ConfigError("requires positive MTP K with padded drafter batch")
    if getattr(cfg.model_config.hf_config, "num_nextn_predict_layers", 1) != 1:
        raise ConfigError("first MTP profile requires a single MTP layer")
    p = cfg.parallel_config
    for field, expected in {"data_parallel_size": 16, "tensor_parallel_size": 1,
                            "pipeline_parallel_size": 1, "prefill_context_parallel_size": 1,
                            "decode_context_parallel_size": 1}.items():
        if getattr(p, field, 1) != expected:
            raise ConfigError(f"requires {field}={expected}")
    if not p.enable_expert_parallel or getattr(p, "use_ubatching", False):
        raise ConfigError("requires EP without microbatching")
    kv = getattr(cfg, "kv_transfer_config", None)
    if kv is None or kv.kv_connector != "MooncakeHybridConnector" or kv.kv_role != "kv_consumer":
        raise ConfigError("requires MooncakeHybridConnector D consumer")
    if (getattr(cfg, "lora_config", None)
            or cfg.cache_config.enable_prefix_caching
            or getattr(cfg.model_config, "enable_return_routed_experts", False)):
        raise ConfigError("LoRA, prefix cache and built-in capture unsupported")
    if cfg.model_config.quantization != "ascend":
        raise ConfigError("requires Ascend W8A8")
    if cfg.model_config.hf_config.architectures != ["DeepseekV4ForCausalLM"]:
        raise ConfigError("requires text-only DeepseekV4ForCausalLM")
    a = runner.ascend_config
    if (getattr(runner, "dynamic_eplb", False) or getattr(runner, "enable_enpu", False)
            or getattr(a.eplb_config, "dynamic_eplb", False)
            or any(getattr(a, k, False) for k in ("enable_force_eplb", "enable_fused_mc2",
               "enable_flashcomm1", "enable_sp_by_pass", "enable_dsa_cp", "enable_shared_expert_dp",
               "multistream_overlap_gate", "multistream_overlap_shared_expert", "mix_placement"))):
        raise ConfigError("EPLB, fused MC2, FlashComm, shared DP, mix placement and multistream unsupported")
    compilation = getattr(a, "ascend_compilation_config", None)
    if compilation and (getattr(compilation, "enable_npugraph_ex", False)
                        or getattr(compilation, "enable_static_kernel", False)):
        raise ConfigError("NPU graph extensions and static kernels unsupported")
    if os.environ.get("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0") != "0":
        raise ConfigError("unset VLLM_ASCEND_ENABLE_FLASHCOMM1 or set it to 0")


class Session:
    def __init__(self, runner, cfg, worker):
        self.cfg, self.worker, self.step = cfg, worker, 0
        self.layers, self.aliases, self.scopes = {}, {}, {}
        self.model_layers = {}
        for scope, model in {"target": runner.model, "mtp": runner.drafter.model}.items():
            discovered = []
            for _, module in model.named_modules():
                name = getattr(module, "layer_name", None)
                if name and getattr(module, "router", None) is not None and hasattr(module, "moe_config"):
                    pattern = r"(?:^|\.)layers\.(\d+)\." if scope == "target" else r"(?:^|\.)mtp\.(\d+)\."
                    match = re.search(pattern, name)
                    key = f"{scope}:{name}"
                    if not match or key in self.layers:
                        raise ConfigError("unknown or duplicate MTP model layer")
                    self.layers[key], self.scopes[key] = int(match[1]), scope
                    self.aliases[id(getattr(module, "routed_experts", module))] = key
                    discovered.append(int(match[1]))
            expected = runner.vllm_config.model_config.hf_config.num_hidden_layers if scope == "target" else 1
            if sorted(discovered) != list(range(expected)):
                raise ConfigError(f"incomplete {scope} MoE layer discovery")
            self.model_layers[scope] = sorted(discovered)
        self.root = Path(cfg["output_root"]) / cfg["run_id"] / worker.worker / "trace"
        create_run(self.root, [worker], self.model_layers, validation_level="a3_unverified_debug")
        self.writer = WorkerWriter(self.root, worker, shard_records=256)
        self.pending, self.verifying, self.last_draft = {}, {}, None
        self.ledger_hashes = {name: hashlib.sha256() for name in ("inputs", "candidates")}
        self.ledger_counts = dict.fromkeys(self.ledger_hashes, 0)
        for name in self.ledger_hashes:
            (self.root.parent / f"{name}.jsonl").touch(exist_ok=False)

    def write_ledger(self, name, value):
        data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
        with (self.root.parent / f"{name}.jsonl").open("ab") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.ledger_hashes[name].update(data)
        self.ledger_counts[name] += 1

    def close(self):
        if not self.writer.failed:
            try:
                if self.verifying:
                    raise IntegrityError("missing verification result at shutdown")
                for name in self.ledger_hashes:
                    atomic_json(self.root.parent / f"{name}-index.json", {
                        "sha256": self.ledger_hashes[name].hexdigest(), "records": self.ledger_counts[name]})
                self.writer.close()
            except BaseException:
                self.writer.fail()
                raise


def get_session(runner, cfg):
    global _session
    if _session is None:
        from vllm.distributed import get_dp_group, get_tp_group, get_ep_group
        dp, tp, ep = get_dp_group(), get_tp_group(), get_ep_group()
        d, t, e = dp.rank_in_group, tp.rank_in_group, ep.rank_in_group
        if (dp.world_size, tp.world_size, ep.world_size) != (16, 1, 16) or e != d or t != 0:
            raise ConfigError("unsupported actual DP/TP/EP topology")
        worker = Worker(PROFILE, cfg["run_id"], cfg["node_id"], f"dp-{d}-tp-{t}", d, t, e, e)
        _session = Session(runner, cfg, worker)
    return _session


def initialize(runner):
    cfg = read_config()
    if cfg:
        check_runtime(runner, runner.vllm_config.compilation_config.cudagraph_mode)
        get_session(runner, cfg)


def engine(session):
    return f"D-v4-mtp-dp-{session.worker.dp_rank}"


@contextmanager
def save_execution(session, rows, scope, input_size):
    state = dict(session=session, rows=rows, input_size=input_size, scope=scope,
                 routes={}, consumed=set(), tensors={})
    token = _active.set(state)
    try:
        yield state
        expected = {name for name, value in session.scopes.items() if value == scope}
        if set(state["routes"]) != expected or state["consumed"] != expected:
            raise IntegrityError("missing MTP/target layer or consumer")
        if any({r.row.input_row for r in values} != set(rows) for values in state["routes"].values()):
            raise IntegrityError("missing or extra input rows at a model layer")
        saved = [r for group in state["routes"].values() for r in group]
        if not session.cfg["save_token_ids"]:
            saved = [replace(r, row=replace(r.row, token_id=None, input_role="redacted")) for r in saved]
        if saved:
            session.writer.declare_execution(scope, list({r.row.input_row:r.row for r in saved}.values()))
            for record in saved:
                session.writer.append(record)
        session.write_ledger("inputs", {"scope": scope, "execution": f"execution-{session.step}",
            "position_semantics": "kernel position; MTP input sequence position = position + 1",
            "rows": [dict(input_row=r.input_row, request=r.local_request, position=r.position,
                         kind=r.kind, candidate_round=r.candidate_round, candidate_node=r.candidate_node)
                     for r in rows.values()]})
    except BaseException:
        session.writer.fail()
        raise
    finally:
        _active.reset(token)


def target_inputs(runner, scheduler, ids, positions, session):
    if ids.ndim != 1 or positions.shape != ids.shape:
        raise IntegrityError("one-dimensional target inputs required")
    actual, pos = ids.detach().cpu().tolist(), positions.detach().cpu().tolist()
    reqs = list(runner.input_batch.req_ids)
    counts = scheduler.num_scheduled_tokens
    drafts = scheduler.scheduled_spec_decode_tokens
    if set(reqs) != set(counts) or len(set(reqs)) != len(reqs) or set(drafts) - set(reqs):
        raise IntegrityError("scheduler differs from target batch")
    if sum(counts.values()) != scheduler.total_num_scheduled_tokens or len(ids) < sum(counts.values()):
        raise IntegrityError("scheduled token count mismatch")
    if session.verifying:
        raise IntegrityError("previous verification has no sampler result")
    rows, offset = {}, 0
    for index, req in enumerate(reqs):
        count = counts[req]
        candidate = drafts.get(req, [])
        if (type(count) is not int or count < 1 or count <= len(candidate)
                or len(candidate) > runner.vllm_config.speculative_config.num_speculative_tokens):
            raise ConfigError("unsupported MTP schedule")
        start = int(runner.input_batch.num_computed_tokens_cpu[index])
        request = runner.requests[req]
        for local in range(count):
            i, position = offset + local, start + local
            if pos[i] != position or actual[i] < 0:
                raise IntegrityError("target actual position/token mismatch")
            candidate_index = local - (count - len(candidate))
            is_candidate = candidate_index >= 0 and bool(candidate)
            pending_list = session.pending.get(req, [])
            pending = pending_list[candidate_index] if is_candidate and candidate_index < len(pending_list) else None
            if is_candidate:
                if (pending is None or actual[i] != candidate[candidate_index] or pending["token_id"] != actual[i]
                        or pending["position"] != position):
                    raise IntegrityError("candidate differs from actual MTP producer")
                session.verifying.setdefault(req, []).append(pending)
                kind, role = "target_verify", "sequence"
                round_id, node = pending["candidate_round"], pending["candidate_node"]
            else:
                if request.get_token_id(position) != actual[i]:
                    raise IntegrityError("target sequence token mismatch")
                kind, role = ("target_verify", "seed") if candidate else ("target_decode", "sequence")
                round_id, node = None, None
            if position >= request.num_prompt_tokens:
                rows[i] = DecodeRow(engine(session), f"execution-{session.step}", session.step,
                    f"{engine(session)}:{req}", req, i, position, actual[i], kind,
                    round_id, node, role)
            elif is_candidate:
                raise IntegrityError("speculative candidate inside prompt unsupported")
        offset += count
    return rows


@contextmanager
def trace_target(runner, scheduler, ids, positions, mode):
    cfg = read_config()
    if cfg is None:
        yield
        return
    check_runtime(runner, mode)
    session = get_session(runner, cfg)
    if _active.get() is not None:
        raise IntegrityError("nested MTP target capture")
    session.step += 1
    try:
        rows = target_inputs(runner, scheduler, ids, positions, session)
        with save_execution(session, rows, "target", len(ids)):
            yield
    except BaseException:
        session.writer.fail()
        raise


_proposing = ContextVar("v4_mtp_proposal", default=None)


def trace_proposal_call(function):
    @wraps(function)
    def wrapped(proposer, *args, **kwargs):
        cfg = read_config()
        if cfg is None or kwargs.get("scheduler_output") is None:
            return function(proposer, *args, **kwargs)
        session = get_session(proposer.runner, cfg)
        k = kwargs.get("num_speculative_tokens")
        maximum = proposer.runner.vllm_config.speculative_config.num_speculative_tokens
        if type(k) is not int or not 0 <= k <= maximum:
            raise ConfigError("scheduled MTP K outside configured range")
        session.last_draft = None
        session.draft_steps, session.draft_step_inputs, session.proposal_k = [], [], k
        session.proposal_recorded = False
        token = _proposing.set(session)
        try:
            result = function(proposer, *args, **kwargs)
            if k and not session.proposal_recorded:
                raise IntegrityError("MTP call has no captured producer/output")
            return result
        except BaseException:
            session.writer.fail()
            raise
        finally:
            _proposing.reset(token)
    return wrapped


@contextmanager
def trace_draft(proposer, model_kwargs, num_tokens, batch_size, sample_indices):
    session = _proposing.get()
    if session is None:
        yield
        return
    runner = proposer.runner
    if (proposer.method != "mtp" or proposer.num_speculative_tokens != session.proposal_k
            or getattr(proposer, "needs_extra_input_slots", False)
            or getattr(proposer, "use_cuda_graph", False)):
        raise ConfigError("unsupported MTP proposer layout")
    if session.last_draft is not None or _active.get() is not None:
        raise IntegrityError("multiple or nested MTP draft forwards")
    session.step += 1
    try:
        ids, positions = model_kwargs["input_ids"], model_kwargs["positions"]
        if ids.ndim != 1 or positions.shape != ids.shape or num_tokens > len(ids):
            raise IntegrityError("invalid actual MTP inputs")
        actual, pos = ids.detach().cpu().tolist(), positions.detach().cpu().tolist()
        reqs = list(runner.input_batch.req_ids)
        starts = proposer.query_start_loc_group[0][:len(reqs)+1].detach().cpu().tolist()
        selected = sample_indices[:len(reqs)].detach().cpu().tolist()
        if (batch_size < len(reqs) or len(selected) != len(reqs) or starts[0] != 0
                or starts[-1] > num_tokens or any(a >= b for a,b in zip(starts, starts[1:]))):
            raise IntegrityError("MTP query metadata differs from actual inputs")
        discarded = set(runner.discard_request_indices.np[:runner.num_discarded_requests].tolist())
        rows, produced = {}, []
        round_id = f"draft-{session.step}"
        session.round_execution = f"execution-{session.step}"
        for index, req in enumerate(reqs):
            start, end, last = starts[index], starts[index+1], selected[index]
            if not start <= last < end:
                raise IntegrityError("MTP sample index outside its request")
            if index in discarded:
                continue
            request = runner.requests[req]
            for i in range(start, last+1):
                if actual[i] < 0 or pos[i] < 0 or request.get_token_id(pos[i]+1) != actual[i]:
                    raise IntegrityError("MTP shifted input differs from request sequence")
                if pos[i]+1 >= request.num_prompt_tokens:
                    rows[i] = DecodeRow(engine(session), f"execution-{session.step}", session.step,
                        f"{engine(session)}:{req}", req, i, pos[i], actual[i], "mtp_draft",
                        input_role="context")
            if last not in rows:
                raise IntegrityError("MTP candidate producer not in Decode input rows")
            produced.append({"request": req, "candidate_round": round_id, "candidate_node": "0",
                "round_execution": session.round_execution,
                "producer_execution": f"execution-{session.step}", "producer_input_row": last,
                "producer_batch_index": index, "producer_kernel_position": pos[last],
                "position": pos[last]+2})
        with save_execution(session, rows, "mtp", len(ids)):
            yield
        session.last_draft = produced
        session.draft_steps.append(produced)
        session.draft_step_inputs.append({})
    except BaseException:
        session.writer.fail()
        raise


@contextmanager
def trace_draft_next(proposer, model_kwargs, batch_size, step, previous_ids, exceeds):
    session = _proposing.get()
    if session is None:
        yield
        return
    session.step += 1
    try:
        if _active.get() is not None or step != len(session.draft_steps) or not 0 < step < session.proposal_k:
            raise IntegrityError("missing, duplicate or nested MTP draft step")
        runner = proposer.runner
        reqs = list(runner.input_batch.req_ids)
        ids, positions = model_kwargs["input_ids"], model_kwargs["positions"]
        if ids.ndim != 1 or positions.shape != ids.shape or len(ids) < len(reqs) or batch_size < len(reqs):
            raise IntegrityError("invalid later MTP input shape")
        actual, pos = ids.detach().cpu().tolist(), positions.detach().cpu().tolist()
        previous = previous_ids.detach().cpu().tolist()
        if any(exceeds[:len(reqs)].detach().cpu().tolist()):
            raise ConfigError("MTP position clamped at model length; capture cannot attribute dummy input")
        first = {item["request"]: item for item in session.draft_steps[0]}
        rows, produced, input_tokens = {}, [], {}
        for index, req in enumerate(reqs):
            if req not in first:  # discarded/prefill request, not a Decode candidate producer
                continue
            base = first[req]
            if (actual[index] != previous[index] or pos[index] != base["producer_kernel_position"] + step
                    or actual[index] < 0):
                raise IntegrityError("later MTP token/position differs from predecessor")
            input_tokens[req] = actual[index]
            rows[index] = DecodeRow(engine(session), f"execution-{session.step}", session.step,
                f"{engine(session)}:{req}", req, index, pos[index], actual[index], "mtp_draft", input_role="context")
            produced.append({"request": req, "candidate_round": base["candidate_round"],
                "candidate_node": str(step), "round_execution": session.round_execution,
                "producer_execution": f"execution-{session.step}", "producer_input_row": index,
                "producer_batch_index": index, "producer_kernel_position": pos[index], "position": pos[index]+2})
        with save_execution(session, rows, "mtp", len(ids)):
            yield
        session.draft_steps.append(produced)
        session.draft_step_inputs.append(input_tokens)
    except BaseException:
        session.writer.fail()
        raise


def record_proposals(proposer, output, sample_indices):
    session = _proposing.get()
    if session is None:
        return output
    try:
        values = output.detach().cpu().tolist()
        if session.proposal_recorded:
            raise IntegrityError("duplicate MTP output hook")
        if (session.last_draft is None or output.ndim != 2 or output.shape[1] != session.proposal_k
                or len(session.draft_steps) != session.proposal_k):
            raise IntegrityError("MTP output lacks every actual producer step")
        if len(values) < len(proposer.runner.input_batch.req_ids):
            raise IntegrityError("MTP output batch too short")
        new_pending = {}
        for step, produced in enumerate(session.draft_steps):
            for item in produced:
                index, req = item["producer_batch_index"], item["request"]
                if session.proposal_k == 1 and int(sample_indices[index]) != item["producer_input_row"]:
                    raise IntegrityError("MTP sampled output order differs from producer")
                if step and session.draft_step_inputs[step][req] != values[index][step-1]:
                    raise IntegrityError("MTP output matrix differs from actual next-step input")
                token_id = values[index][step]
                if type(token_id) is not int or token_id < 0:
                    raise IntegrityError("invalid MTP output token")
                pending = dict(item, token_id=token_id, proposal_k=session.proposal_k)
                new_pending.setdefault(req, []).append(pending)
                saved = dict(pending)
                if not session.cfg["save_token_ids"]:
                    saved["token_id"] = None
                session.write_ledger("candidates", saved)
        session.pending.update(new_pending)
        for req in set(session.pending) - set(proposer.runner.requests):
            del session.pending[req]
        session.proposal_recorded = True
    except BaseException:
        session.writer.fail()
        raise
    return output


def record_verification(runner, metadata, sampler_output):
    cfg = read_config()
    if cfg is None:
        return
    session = get_session(runner, cfg)
    try:
        reqs = list(runner.input_batch.req_ids)
        lengths = metadata.num_draft_tokens
        samples = sampler_output.sampled_token_ids.detach().cpu().tolist()
        draft_ids = metadata.draft_token_ids.detach().cpu().tolist()
        if len(lengths) != len(reqs) or len(samples) < len(reqs):
            raise IntegrityError("sampler batch differs from verification")
        offset = 0
        expected = {req for req, k in zip(reqs, lengths) if k}
        if expected != set(session.verifying):
            raise IntegrityError("sampler candidates differ from executed verification")
        for req, k, sampled in zip(reqs, lengths, samples):
            if not k:
                continue
            candidates = session.verifying[req]
            if k != len(candidates) or draft_ids[offset:offset+k] != [c["token_id"] for c in candidates]:
                raise IntegrityError("sampler draft token differs from MTP candidate")
            offset += k
            valid = [v for v in sampled if v >= 0]
            if not 1 <= len(valid) <= k+1 or sampled[:len(valid)] != valid or any(v < -1 for v in sampled):
                raise IntegrityError("invalid rejection sampler output layout")
            accepted_count = len(valid)-1
            if valid[:accepted_count] != [c["token_id"] for c in candidates[:accepted_count]]:
                raise IntegrityError("accepted prefix differs from candidate")
            for node, candidate in enumerate(candidates):
                session.writer.append(CandidateResult(f"{engine(session)}:{req}", engine(session),
                    candidate["candidate_round"], candidate["candidate_node"], node < accepted_count))
            session.pending.pop(req, None)
        if offset != len(draft_ids):
            raise IntegrityError("extra sampler draft IDs")
        session.verifying.clear()
    except BaseException:
        session.writer.fail()
        raise

def capture_routes(layer, ids, weights, prepare, extra, context, num_experts, forbidden=False):
    state = _active.get()
    if state is None:
        return
    import torch
    session = state["session"]
    name = session.aliases.get(id(layer))
    if (forbidden or not name or session.scopes.get(name) != state["scope"] or name in state["routes"] or ids.ndim != 2
            or ids.dtype not in (torch.int32, torch.int64) or ids.shape != weights.shape):
        raise IntegrityError("invalid logical routing path")
    if (getattr(layer, "mix_placement", False) or getattr(layer, "dynamic_eplb", False)
            or getattr(layer.router, "bias_vl", None) is not None):
        raise ConfigError("mixed/shared expert placement, EPLB and vision routing unsupported")
    if (getattr(extra, "flash_comm_v1_enabled", False) or getattr(prepare, "replace_allreduce", False)
            or getattr(prepare, "enable_shared_expert_dp", False)):
        raise ConfigError("SP and shared-DP prepare unsupported in V4 debug profile")
    meta = getattr(context, "dp_metadata", None)
    dp_sizes = meta.num_tokens_across_dp_cpu.tolist() if meta else None
    size = state["input_size"]
    if type(prepare).__name__ == "PrepareAndFinalizeWithAllGather" and dp_sizes:
        # Forward padding may exceed actual local DP tokens. Only real rows are mapped.
        size = dp_sizes[session.worker.dp_rank]
        if any(i >= size for i in state["rows"]):
            raise IntegrityError("DP metadata excludes a valid input")
    indices = selector_rows(type(prepare).__name__, input_size=size, valid_indices=state["rows"],
        tp_size=1, tp_rank=session.worker.tp_rank, dp_rank=session.worker.dp_rank,
        output_size=len(ids), padded_size=getattr(extra, "padded_num_tokens", None), dp_sizes=dp_sizes) if state["rows"] else []
    index = [j for _, j in indices]
    selected = ids[index].detach().cpu()
    scales = weights[index].detach().float().cpu() if session.cfg["save_expert_weights"] else None
    state["routes"][name] = [Route(session.worker, session.worker, state["rows"][i], state["scope"],
        session.layers[name], int(num_experts), tuple(selected[k].tolist()),
        tuple(scales[k].tolist()) if scales is not None else None) for k, (i, _) in enumerate(indices)]
    expected = selected.clone()
    mapping = getattr(layer, "log2phy", None)
    if mapping is not None and index:
        expected = mapping[ids[index]].detach().cpu().clone()
    state["tensors"][name] = (index, expected)


def capture_consumer(layer, ids):
    state = _active.get()
    if state is None:
        return
    import torch
    name = state["session"].aliases.get(id(layer))
    if name not in state["tensors"] or name in state["consumed"]:
        raise IntegrityError("consumer without unique selector")
    index, selected = state["tensors"][name]
    if ids.ndim != 2 or any(j >= len(ids) for j in index) or not torch.equal(ids[index].detach().cpu(), selected):
        raise IntegrityError("consumer experts differ from captured expected mapping")
    state["consumed"].add(name)


def close():
    global _session
    if _session:
        if not _session.writer.failed:
            _session.close()
        _session = None


