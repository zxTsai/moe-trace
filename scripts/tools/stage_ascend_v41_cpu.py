"""Create a hash-checked V4.1 CPU-hook source copy. NOT an A3 deployment tool."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HOOK = "moe_trace.adapters.ascend_v41_source"
EDITS = {
    "vllm_ascend/worker/model_runner_v1.py": [
        ("        assert forward_context is not None\n        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:",
         "        assert forward_context is not None\n"
         f"        from {HOOK} import verify_forward\n"
         "        verify_forward(input_ids, positions, forward_context.cudagraph_runtime_mode)\n"
         "        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:")],
    "vllm_ascend/ops/fused_moe/routed_experts.py": [
        ("        if self.log2phy is not None:\n            topk_ids = self.log2phy[topk_ids]",
         f"        from {HOOK} import logical_routes\n"
         "        logical_routes(self, topk_ids, topk_weights,\n"
         "                       enable_force_load_balance or get_ascend_config().enable_force_eplb)\n"
         "        if self.log2phy is not None:\n            topk_ids = self.log2phy[topk_ids]")],
    "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py": [
        ("        assert topk_weights is not None\n\n        activation = getattr(layer, \"activation\", \"silu\")",
         "        assert topk_weights is not None\n"
         f"        from {HOOK} import consumer_routes\n"
         "        consumer_routes(layer, topk_ids, topk_weights)\n\n"
         "        activation = getattr(layer, \"activation\", \"silu\")")],
}


def transform(name, text):
    for old, new in EDITS[name]:
        if text.count(old) != 1:
            raise ValueError(f"nonunique source anchor: {name}")
        text = text.replace(old, new)
    ast.parse(text, filename=name)
    return text


def stage(vllm_root, ascend_root, output_parent):
    lock = json.loads((ROOT / "docs/design/ascend-v41-source-lock.json").read_text())
    check = runpy.run_path(str(ROOT / "scripts/tools/check_ascend_v41_sources.py"))["check_sources"]
    check(vllm_root, ascend_root, lock)
    parent = Path(output_parent).resolve()
    for root in (Path(vllm_root).resolve(), Path(ascend_root).resolve()):
        if parent == root or parent.is_relative_to(root):
            raise ValueError("output must not be inside input sources")
    sources = {name: (Path(ascend_root) / name).read_text() for name in EDITS}
    modified = {name: transform(name, text) for name, text in sources.items()}
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="ascend-v41-cpu-", dir=parent))
    shutil.copytree(Path(ascend_root) / "vllm_ascend", output / "vllm_ascend",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patch = ""
    for name, text in modified.items():
        (output / name).write_text(text, encoding="utf-8")
        patch += "".join(difflib.unified_diff(sources[name].splitlines(True), text.splitlines(True),
                                            fromfile=f"a/{name}", tofile=f"b/{name}"))
    (output / "hooks.patch").write_text(patch, encoding="utf-8")
    (output / "source-hooks.json").write_text(json.dumps({
        "validation_level": "cpu_source_hooks", "npu_verified": False,
        "base_commits": lock["base_commits"],
        "patched_sha256": {name: hashlib.sha256(text.encode()).hexdigest()
                           for name, text in modified.items()},
    }, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--ascend-root", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    args = parser.parse_args()
    print(stage(args.vllm_root, args.ascend_root, args.output_parent))


if __name__ == "__main__":
    main()
