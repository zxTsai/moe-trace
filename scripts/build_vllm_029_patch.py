"""Regenerate the minimal patch from unmodified installed 0.29.0 sources."""
import difflib
import sys
from pathlib import Path

root = Path(sys.argv[1])
edits = {
    "v1/engine/core.py": [(
        "        # Exit when no work remaining\n        if not self.has_work():\n",
        "        # Exit when no work remaining\n        if not self.has_work():\n"
        "            from moe_trace.adapters.vllm_0_29_0 import close_trace\n"
        "            close_trace()\n")],
    "v1/worker/gpu_model_runner.py": [(
        "        with (\n            set_forward_context(\n                attn_metadata,",
        "        from moe_trace.adapters.vllm_0_29_0 import trace_execution\n"
        "        with (\n"
        "            trace_execution(self, scheduler_output, num_scheduled_tokens_np,\n"
        "                            input_ids, positions),\n"
        "            set_forward_context(\n                attn_metadata,")],
    "model_executor/layers/fused_moe/runner/moe_runner.py": [(
        "        if self.routed_experts.quant_method.is_monolithic:\n            # Monolithic kernels:",
        "        from moe_trace.adapters.vllm_0_29_0 import check_route_backend, record_routes\n"
        "        check_route_backend(self.routed_experts.quant_method.is_monolithic)\n"
        "        if self.routed_experts.quant_method.is_monolithic:\n            # Monolithic kernels:"), (
        "            fused_out = self.routed_experts.forward_modular(",
        "            record_routes(self.layer_name, topk_ids, topk_weights)\n"
        "            fused_out = self.routed_experts.forward_modular(")],
    "v1/core/sched/scheduler.py": [(
        "            self.requests[request.request_id] = request\n",
        "            self.requests[request.request_id] = request\n"
        "            from moe_trace.adapters.vllm_0_29_0 import request_started\n"
        "            request_started(self, request)\n"), (
        "        assert request.is_finished()\n\n        self._inflight_prefills.discard(request)",
        "        assert request.is_finished()\n"
        "        from moe_trace.adapters.vllm_0_29_0 import request_finished\n"
        "        request_finished(self, request)\n\n        self._inflight_prefills.discard(request)")],
    "v1/executor/uniproc_executor.py": [(
        "    def shutdown(self) -> None:\n        if worker := self.driver_worker:",
        "    def shutdown(self) -> None:\n"
        "        from moe_trace.adapters.vllm_0_29_0 import close_trace\n"
        "        close_trace()\n        if worker := self.driver_worker:")],
}
for name, replacements in edits.items():
    before = (root / name).read_text()
    after = before
    for old, new in replacements:
        if after.count(old) != 1:
            raise RuntimeError(f"nonunique patch anchor in {name}")
        after = after.replace(old, new)
    sys.stdout.writelines(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{name}", tofile=f"b/{name}"))
