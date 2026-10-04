"""Opt-in, small-run reference probes; never used for ordinary collection."""
from __future__ import annotations

import contextvars
import inspect
import json
from pathlib import Path

_identity = contextvars.ContextVar("trace_reference_identity", default=None)
_layer = contextvars.ContextVar("trace_reference_layer", default=None)


def install(runner, directory):
    if getattr(runner, "_moe_reference_installed", False):
        return
    runner._moe_reference_installed = True
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    def emit(record):
        identity = _identity.get()
        if identity is None:
            return
        with (path / "reference.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**identity, **record}) + "\n")

    def model_input(module, args, kwargs):
        if _identity.get() is None:
            return
        # Read the actual GPU arguments, attention row boundaries, and independent
        # cached request token history. Do not use collector InputRow objects.
        ids = kwargs["input_ids"].detach().cpu().tolist()
        positions = kwargs["positions"].detach().cpu().tolist()
        boundaries = runner.query_start_loc_cpu[:runner.input_batch.num_reqs + 1].tolist()
        rows = []
        for index, req_id in enumerate(runner.input_batch.req_ids):
            request = runner.requests[req_id]
            for row in range(boundaries[index], boundaries[index + 1]):
                position = positions[row]
                expected_token = request.get_token_id(position)
                if ids[row] != expected_token:
                    raise RuntimeError(f"reference input mismatch: {req_id}, {position}")
                rows.append(dict(row_id=row, request_id=req_id,
                                 token_position=position, token_id=ids[row],
                                 phase="prefill" if position < request.num_prompt_tokens else "decode"))
        if len(rows) != len(ids):
            raise RuntimeError("reference input includes padding or unmapped rows")
        emit(dict(kind="inputs", rows=rows))

    runner.model.register_forward_pre_hook(model_input, with_kwargs=True)

    def wrap_consumer(original, name):
        signature = inspect.signature(original.forward if hasattr(original, "forward") else original)

        def wrapped(*args, **kwargs):
            if _identity.get() is not None:
                bound = signature.bind(*args, **kwargs).arguments
                emit(dict(kind="consumer", layer_name=_layer.get(), consumer=name,
                          expert_ids=bound["topk_ids"].detach().cpu().tolist(),
                          expert_weights=bound["topk_weights"].detach().float().cpu().tolist()))
            return original(*args, **kwargs)
        return wrapped

    import vllm.model_executor.layers.fused_moe.layer as implementation
    implementation.fused_experts = wrap_consumer(implementation.fused_experts, "fused_experts")
    for module in runner.model.modules():
        if not isinstance(module, implementation.FusedMoE):
            continue
        method = module.quant_method
        original = method.forward_cuda

        def scoped(*args, _original=original, _name=module.layer_name, **kwargs):
            token = _layer.set(_name)
            try:
                return _original(*args, **kwargs)
            finally:
                _layer.reset(token)

        method.forward_cuda = scoped
        # CustomOp cached the selected bound forward during initialization.
        method._forward_method = scoped
        if method.fused_experts is not None:
            method.fused_experts = wrap_consumer(method.fused_experts, "modular_experts")


def begin(run_id, execution_id):
    return _identity.set(dict(run_id=run_id, execution_id=execution_id))


def end(token):
    _identity.reset(token)
