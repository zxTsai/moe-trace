"""Opt-in independent forward-input and expert-consumer probes for 0.29.0."""
import contextvars
import inspect
import json
from pathlib import Path

_identity = contextvars.ContextVar("reference_029_identity", default=None)


def begin(run_id, execution_id):
    return _identity.set(dict(run_id=run_id, execution_id=execution_id))


def end(token):
    _identity.reset(token)


def install(runner, directory):
    if getattr(runner, "_moe_reference_029_installed", False):
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    def emit(record):
        identity = _identity.get()
        if identity is not None:
            with (path / "reference.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**identity, **record}) + "\n")

    def model_input(module, args, kwargs):
        if _identity.get() is None:
            return
        ids = kwargs["input_ids"].detach().cpu().tolist()
        positions = kwargs["positions"].detach().cpu().tolist()
        boundaries = runner.query_start_loc.gpu[:runner.input_batch.num_reqs + 1].cpu().tolist()
        rows = []
        for index, req_id in enumerate(runner.input_batch.req_ids):
            request = runner.requests[req_id]
            for row in range(boundaries[index], boundaries[index + 1]):
                pos = positions[row]
                if ids[row] != request.get_token_id(pos):
                    raise RuntimeError(f"reference input mismatch: {req_id}, {pos}")
                rows.append(dict(row_id=row, request_id=req_id, token_position=pos,
                                 token_id=ids[row],
                                 phase="prefill" if pos < request.num_prompt_tokens else "decode"))
        if len(rows) != len(ids):
            raise RuntimeError("reference input includes unmapped rows")
        emit(dict(kind="inputs", rows=rows))

    runner.model.register_forward_pre_hook(model_input, with_kwargs=True)
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    for module in runner.model.modules():
        if not isinstance(module, MoERunner):
            continue
        original = module.routed_experts.forward_modular
        signature = inspect.signature(original)

        def consumer(*args, _original=original, _signature=signature,
                     _name=module.layer_name, **kwargs):
            if _identity.get() is not None:
                bound = _signature.bind(*args, **kwargs).arguments
                emit(dict(kind="consumer", layer_name=_name, consumer="forward_modular",
                          expert_ids=bound["topk_ids"].detach().cpu().tolist(),
                          expert_weights=bound["topk_weights"].detach().float().cpu().tolist()))
            return _original(*args, **kwargs)

        module.routed_experts.forward_modular = consumer
    runner._moe_reference_029_installed = True
