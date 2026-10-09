"""Copy pinned Ascend sources and add Qwen capture hooks; original package is unchanged."""
import argparse
import ast
import difflib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

from inspect_qwen36_ascend import inspect_sources

HOOK = "moe_trace.adapters.ascend_qwen36"
EDITS = {
    "worker/model_runner_v1.py": [(
        "            hidden_states = self._model_forward(\n"
        "                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs\n"
        "            )",
        f"            from {HOOK} import trace_target\n"
        "            with trace_target(self, scheduler_output, input_ids, positions, cudagraph_mode):\n"
        "                hidden_states = self._model_forward(\n"
        "                    num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs\n"
        "                )")],
    "quantization/methods/w8a8_dynamic.py": [(
        "from vllm.logger import logger\n",
        "from vllm.logger import logger\nfrom vllm.forward_context import get_forward_context\n"), (
        "        assert topk_ids is not None\n        assert topk_weights is not None\n",
        "        assert topk_ids is not None\n        assert topk_weights is not None\n"
        f"        from {HOOK} import capture_routes\n"
        "        capture_routes(layer, topk_ids, topk_weights,\n"
        "            _EXTRA_CTX.moe_comm_method.prepare_finalize, _EXTRA_CTX,\n"
        "            get_forward_context(), num_logical_experts,\n"
        "            bool(enable_force_load_balance or self.dynamic_eplb or\n"
        "                 self.multistream_overlap_gate or mix_placement or tid2eid is not None or zero_expert_num))\n"), (
        "        final_hidden_states = moe_comm_method.fused_experts(\n",
        f"        from {HOOK} import capture_consumer\n"
        "        capture_consumer(layer, topk_ids)\n"
        "        final_hidden_states = moe_comm_method.fused_experts(\n")],
    "worker/worker.py": [(
        "            self.model_runner.load_model()\n",
        "            self.model_runner.load_model()\n"
        f"            from {HOOK} import initialize\n            initialize(self.model_runner)\n"), (
        "    def shutdown(self) -> None:\n",
        f"    def shutdown(self) -> None:\n        from {HOOK} import close\n        close()\n")],
}


def transform(name, source):
    for old, new in EDITS[name]:
        if source.count(old) != 1:
            raise ValueError(f"missing or ambiguous hook anchor: {name}")
        source = source.replace(old, new)
    ast.parse(source)
    return source


def stage(package, parent):
    package, parent = Path(package).resolve(), Path(parent).resolve()
    if parent == package or parent.is_relative_to(package):
        raise ValueError("copy must be outside original package")
    inventory = inspect_sources(package)
    if not inventory["all_reference_files_match"]:
        raise ValueError("Ascend source mismatch; do not force the patch")
    original = {name: (package / name).read_text() for name in EDITS}
    changed = {name: transform(name, content) for name, content in original.items()}
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="qwen36-overlay-", dir=parent))
    shutil.copytree(package, output / "vllm_ascend", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    import hashlib
    for name, source in changed.items():
        (output / "vllm_ascend" / name).write_text(source)
    patch = "".join("".join(difflib.unified_diff(original[name].splitlines(True), content.splitlines(True),
        fromfile=f"a/vllm_ascend/{name}", tofile=f"b/vllm_ascend/{name}")) for name, content in changed.items())
    (output / "hooks.patch").write_text(patch)
    (output / "overlay.json").write_text(json.dumps({"profile": "qwen36-single-node-debug",
        "npu_verified": False, "ascend_commit": inventory["reference_commit"],
        "patched_sha256": {name: hashlib.sha256(content.encode()).hexdigest()
                           for name, content in changed.items()}}, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascend-package", type=Path, help="Default: detect current Python package")
    parser.add_argument("--output-parent", type=Path, required=True)
    args = parser.parse_args()
    package = args.ascend_package
    if package is None:
        spec = importlib.util.find_spec("vllm_ascend")
        if spec is None or not spec.origin:
            parser.error("vllm_ascend not installed in current Python")
        package = Path(spec.origin).parent
    print(stage(package, args.output_parent))


if __name__ == "__main__":
    main()
