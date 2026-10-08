"""Stage eager A3 debug hooks in a copy; no installation or service startup."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HOOK = "moe_trace.adapters.ascend_v41_debug"
EDITS = {
    "vllm_ascend/worker/model_runner_v1.py": [
        ("            hidden_states = self._model_forward(\n"
         "                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs\n"
         "            )",
         f"            from {HOOK} import trace_target\n"
         "            with trace_target(self, scheduler_output, input_ids, positions, cudagraph_mode):\n"
         "                hidden_states = self._model_forward(\n"
         "                    num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs\n"
         "                )")],
    "vllm_ascend/ops/fused_moe/routed_experts.py": [
        ("        if self.log2phy is not None:\n            topk_ids = self.log2phy[topk_ids]",
         f"        from {HOOK} import capture_routes\n"
         "        capture_routes(self, topk_ids, topk_weights,\n"
         "                       _EXTRA_CTX.moe_comm_method.prepare_finalize, _EXTRA_CTX, get_forward_context(),\n"
         "                       get_moe_num_logical_experts(self, self.moe_config.num_experts,\n"
         "                           global_redundant_expert_num=self.global_redundant_expert_num,\n"
         "                           num_shared_experts=self.n_shared_experts or 0),\n"
         "                       enable_force_load_balance or get_ascend_config().enable_force_eplb)\n"
         "        if self.log2phy is not None:\n            topk_ids = self.log2phy[topk_ids]")],
    "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py": [
        ("        assert topk_weights is not None\n\n        activation = getattr(layer, \"activation\", \"silu\")",
         "        assert topk_weights is not None\n"
         f"        from {HOOK} import capture_consumer\n"
         "        capture_consumer(layer, topk_ids, topk_weights)\n\n"
         "        activation = getattr(layer, \"activation\", \"silu\")")],
    "vllm_ascend/worker/worker.py": [
        ("    def shutdown(self) -> None:\n        if ensure_kv_transfer_shutdown is not None:",
         "    def shutdown(self) -> None:\n"
         f"        from {HOOK} import close_debug\n"
         "        close_debug()\n        if ensure_kv_transfer_shutdown is not None:")],
}


def transform(name, source):
    for old, new in EDITS[name]:
        if source.count(old) != 1:
            raise ValueError(f"missing/nonunique debug anchor: {name}")
        source = source.replace(old, new)
    ast.parse(source, filename=name)
    return source


def check_sources(vllm_package, ascend_package):
    lock = json.loads((ROOT / "patches/ascend/deepseek-v41-debug/base_files.json").read_text())
    roots = {"vllm": Path(vllm_package), "vllm_ascend": Path(ascend_package)}
    for name, expected in lock["sha256"].items():
        package, relative = name.split("/", 1)
        data = (roots[package] / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"debug source hash mismatch: {name}")
    for name in EDITS:
        transform(name, (roots["vllm_ascend"] / name.split("/", 1)[1]).read_text())
    return lock


def stage(vllm_package, ascend_package, parent):
    lock = check_sources(vllm_package, ascend_package)
    parent = Path(parent).resolve()
    for source in (Path(vllm_package).resolve(), Path(ascend_package).resolve()):
        if parent == source or parent.is_relative_to(source):
            raise ValueError("debug copy cannot be inside source packages")
    sources = {name: (Path(ascend_package) / name.split("/", 1)[1]).read_text() for name in EDITS}
    changed = {name: transform(name, source) for name, source in sources.items()}
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="ascend-v41-debug-", dir=parent))
    shutil.copytree(ascend_package, output / "vllm_ascend", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patch = ""
    for name, source in changed.items():
        (output / name).write_text(source, encoding="utf-8")
        patch += "".join(difflib.unified_diff(sources[name].splitlines(True), source.splitlines(True),
                                            fromfile=f"a/{name}", tofile=f"b/{name}"))
    (output / "debug-hooks.patch").write_text(patch)
    (output / "debug-overlay.json").write_text(json.dumps({
        "profile": "1P1D-debug", "npu_verified": False, "base_commits": lock["base_commits"],
        "patched_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in changed.items()},
    }, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-package", type=Path, required=True)
    parser.add_argument("--ascend-package", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print(json.dumps({"source_check": "passed", "base_commits": check_sources(args.vllm_package, args.ascend_package)["base_commits"]}))
    elif args.output_parent is None:
        parser.error("--output-parent is required when creating a copy")
    else:
        print(stage(args.vllm_package, args.ascend_package, args.output_parent))


if __name__ == "__main__":
    main()
