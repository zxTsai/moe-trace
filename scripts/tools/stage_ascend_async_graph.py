"""Create a checked source overlay for the existing A3 container (no pip changes)."""
import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "patches/ascend/deepseek-v4-023post1-async-graph/reference-files.json"
HOOK = "moe_trace.adapters.ascend_023_debug"
EDITS = {
    "compilation/acl_graph.py": [
        ("import dataclasses\n", "import dataclasses\nfrom moe_trace.adapters.ascend_graph_bridge import trace_acl_graph\n"),
        ("    def __call__(self, *args, **kwargs):\n", "    @trace_acl_graph\n    def __call__(self, *args, **kwargs):\n"),
    ],
    "quantization/methods/w8a8_dynamic.py": [
        ("        assert topk_ids is not None\n        assert topk_weights is not None\n",
         "        assert topk_ids is not None\n        assert topk_weights is not None\n"
         f"        from {HOOK} import capture_routes\n"
         "        capture_routes(layer, topk_ids, topk_weights,\n"
         "                       _EXTRA_CTX.moe_comm_method.prepare_finalize, num_logical_experts,\n"
         "                       enable_force_load_balance or self.dynamic_eplb)\n"),
    ],
    "worker/worker.py": [
        ("            self.model_runner.load_model()\n", "            self.model_runner.load_model()\n"
         f"            from {HOOK} import initialize\n            initialize(self.model_runner)\n"),
        ("    def shutdown(self) -> None:\n", "    def shutdown(self) -> None:\n"
         f"        from {HOOK} import close\n        close()\n"),
    ],
}
# Keep imports at module level for decorators, outside the graph's device work.
EDITS["worker/model_runner_v1.py"] = [
    ("            hidden_states = self._model_forward(\n"
     "                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs\n"
     "            )",
     f"            from {HOOK} import trace_target\n"
     "            with trace_target(self, scheduler_output, input_ids, positions, cudagraph_mode):\n"
     "                hidden_states = self._model_forward(\n"
     "                    num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs\n"
     "                )"),
    ("    def _sample(self, logits, spec_decode_metadata):\n",
     f"    from {HOOK} import trace_sample as _moe_trace_sample\n\n"
     "    @_moe_trace_sample\n    def _sample(self, logits, spec_decode_metadata):\n"),
]
EDITS["spec_decode/llm_base_proposer.py"] = [
    ("    def _propose(\n", f"    from {HOOK} import trace_proposal as _moe_trace_proposal\n\n"
     "    @_moe_trace_proposal\n    def _propose(\n"),
    ("        ret_hidden_states = self.model(**model_kwargs)\n",
     f"        from {HOOK} import trace_mtp_forward\n"
     "        with trace_mtp_forward(self, model_kwargs, num_tokens, batch_size, token_indices_to_sample, 0):\n"
     "            ret_hidden_states = self.model(**model_kwargs)\n"),
    ("            ret_hidden_states = self.model(**model_kwargs)\n            if not self.model_returns_tuple():",
     f"            from {HOOK} import trace_mtp_forward\n"
     "            with trace_mtp_forward(self, model_kwargs, input_batch_size, batch_size, token_indices_to_sample, draft_index + 1):\n"
     "                ret_hidden_states = self.model(**model_kwargs)\n            if not self.model_returns_tuple():"),
]


def transform(name, source):
    for old, new in EDITS[name]:
        # Match whole lines; an eight-space anchor must not match a twelve-space line.
        pattern = re.compile("^" + re.escape(old), re.MULTILINE)
        if len(list(pattern.finditer(source))) != 1:
            raise ValueError(f"missing/nonunique source anchor: {name}: {old[:75]!r}")
        source = pattern.sub(lambda match: new, source)
    ast.parse(source, filename=name)
    return source


def locate(package, explicit):
    if explicit:
        return Path(explicit).resolve()
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise ValueError(f"{package} not found in service Python; specify package path")
    return Path(spec.origin).resolve().parent


def stage(vllm_package, ascend_package, parent, check_only=False):
    roots = {"vllm": Path(vllm_package).resolve(), "vllm_ascend": Path(ascend_package).resolve()}
    reference = json.loads(REFERENCE.read_text())
    for name, digest in reference["sha256"].items():
        package, relative = name.split("/", 1)
        path = roots[package] / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"source mismatch: {path}; original sources were not changed")
    originals = {name: (roots["vllm_ascend"] / name).read_text() for name in EDITS}
    changed = {name: transform(name, text) for name, text in originals.items()}
    if check_only:
        return {"source_check": "passed", "changed_files": list(changed), "npu_verified": False}
    parent = Path(parent).resolve()
    for source in roots.values():
        if parent == source or parent.is_relative_to(source):
            raise ValueError("overlay must be outside the installed source packages")
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="ascend-023-async-", dir=parent))
    try:
        shutil.copytree(roots["vllm_ascend"], output / "vllm_ascend",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
        patch = []
        for name, text in changed.items():
            (output / "vllm_ascend" / name).write_text(text)
            patch.extend(difflib.unified_diff(originals[name].splitlines(True), text.splitlines(True),
                         fromfile=f"a/vllm_ascend/{name}", tofile=f"b/vllm_ascend/{name}"))
        (output / "debug-hooks.patch").write_text("".join(patch))
        hashes = {}
        for name in reference["sha256"]:
            package, relative = name.split("/", 1)
            path = output / name if package == "vllm_ascend" else roots[package] / relative
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / "overlay.json").write_text(json.dumps({
            "profile": reference["profile"], "packages": {k: str(v) for k,v in roots.items()},
            "sha256": hashes, "debug_entry_ready": True, "npu_verified": False,
            "plugin_sha256": {name: hashlib.sha256((ROOT/"src"/name).read_bytes()).hexdigest()
                for name in ("moe_trace/adapters/ascend_023_debug.py", "moe_trace/adapters/ascend_graph_bridge.py",
                             "moe_trace/adapters/graph_routes.py", "moe_trace/adapters/ascend_load_summary.py", "moe_trace/io.py")},
        }, ensure_ascii=False, indent=2)+"\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-package", type=Path)
    parser.add_argument("--ascend-package", type=Path)
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and args.output_parent is None:
        parser.error("--output-parent is required")
    try:
        result = stage(locate("vllm", args.vllm_package), locate("vllm_ascend", args.ascend_package),
                       args.output_parent, args.check)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result) if isinstance(result, dict) else result)


if __name__ == "__main__":
    main()
