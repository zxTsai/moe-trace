"""Standard-library-only, hash-checked Ascend source overlays (no installation)."""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

HOOK = "moe_trace.adapters.ascend_vllm_0_23_0"
BASES = {"vllm": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
         "vllm_ascend": "5cb98caaadeff42b5b62b996e34bb2aaa29d20fd"}
EDITS = {
    "vllm/v1/core/sched/scheduler.py": [
        ("            self.requests[request.request_id] = request\n",
         "            self.requests[request.request_id] = request\n"
         f"            from {HOOK} import request_started\n"
         "            request_started(self, request)\n"),
        ("        assert request.is_finished()\n\n        self._inflight_prefills.discard(request)",
         "        assert request.is_finished()\n"
         f"        from {HOOK} import request_finished\n"
         "        request_finished(self, request)\n\n        self._inflight_prefills.discard(request)"),
        ("    def shutdown(self) -> None:\n        logger.debug_once",
         "    def shutdown(self) -> None:\n"
         f"        from {HOOK} import close_trace\n"
         "        close_trace()\n        logger.debug_once")],
    "vllm_ascend/worker/model_runner_v1.py": [
        ('        with (\n            record_function_or_nullcontext("forward"),',
         f"        from {HOOK} import trace_execution\n"
         "        with (\n"
         "            trace_execution(self, scheduler_output, input_ids, positions),\n"
         '            record_function_or_nullcontext("forward"),')],
    "vllm_ascend/ops/fused_moe/fused_moe.py": [
        ("        final_hidden_states = moe_comm_method.fused_experts(\n",
         f"        from {HOOK} import record_routes\n"
         "        record_routes(layer.layer_name, topk_ids, topk_weights,\n"
         "                      zero_experts=zero_expert_num,\n"
         "                      force_load_balance=enable_force_load_balance,\n"
         "                      dynamic_eplb=self.dynamic_eplb,\n"
         "                      comm_type=_EXTRA_CTX.moe_comm_type.name)\n"
         "        final_hidden_states = moe_comm_method.fused_experts(\n")],
}


def transform(name, before):
    after = before
    for old, new in EDITS[name]:
        if after.count(old) != 1:
            raise ValueError(f"nonunique/absent source anchor: {name}")
        after = after.replace(old, new)
    ast.parse(after, filename=name)
    return after


def collect_sources(vllm_root, ascend_root):
    roots = {"vllm": Path(vllm_root), "vllm_ascend": Path(ascend_root)}
    return {name: (roots[name.split("/", 1)[0]] / name.split("/", 1)[1]).read_bytes()
            for name in EDITS}


def generate(vllm_root, ascend_root, output):
    """Mechanical regeneration from the pinned clean package sources."""
    for name, root in (("vllm", vllm_root), ("vllm_ascend", ascend_root)):
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if actual != BASES[name]:
            raise ValueError(f"patch generation requires pinned clean commit: {name}")
        files = [n.split("/", 1)[1] for n in EDITS if n.startswith(name + "/")]
        subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", *files],
                       check=True, capture_output=True)
    sources = collect_sources(vllm_root, ascend_root)
    manifest = {"base_commits": BASES, "sha256": {
        name: hashlib.sha256(data).hexdigest() for name, data in sources.items()},
        "patched_sha256": {name: hashlib.sha256(transform(name, data.decode("utf-8")).encode()).hexdigest()
                           for name, data in sources.items()}}
    chunks = []
    for name, data in sources.items():
        before = data.decode("utf-8")
        after = transform(name, before)
        chunks.extend(difflib.unified_diff(before.splitlines(keepends=True),
                                          after.splitlines(keepends=True),
                                          fromfile=f"a/{name}", tofile=f"b/{name}"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "base_files.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "0001-trace-hooks.patch").write_text("".join(chunks))
    return manifest


def check_sources(vllm_root, ascend_root, manifest):
    if manifest.get("base_commits") != BASES or set(manifest.get("sha256", {})) != set(EDITS):
        raise ValueError("unsupported Ascend manifest")
    sources = collect_sources(vllm_root, ascend_root)
    for name, data in sources.items():
        if hashlib.sha256(data).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"source version/hash mismatch (possibly already patched): {name}")
        transform(name, data.decode("utf-8"))
    return sources


def stage(vllm_root, ascend_root, manifest, output_parent):
    sources = check_sources(vllm_root, ascend_root, manifest)
    # Never write to the caller's installed package or overwrite an existing overlay.
    parent = Path(output_parent)
    for source in (Path(vllm_root).resolve(), Path(ascend_root).resolve()):
        if parent.resolve() == source or source in parent.resolve().parents:
            raise ValueError("overlay output must not be inside a source package")
    parent.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix="moe-ascend-023-", dir=parent))
    for name, source in (("vllm", vllm_root), ("vllm_ascend", ascend_root)):
        shutil.copytree(source, target / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, data in sources.items():
        (target / name).write_text(transform(name, data.decode("utf-8")), encoding="utf-8")
    return target


def check_overlay(overlay):
    manifest_path = Path(__file__).resolve().parents[3] / "patches/ascend/0.23.0/base_files.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["patched_sha256"].items():
        data = (Path(overlay) / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"patched source hash mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-package", type=Path, required=True)
    parser.add_argument("--ascend-package", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[3]
                        / "patches/ascend/0.23.0/base_files.json")
    parser.add_argument("--check", action="store_true", help="read-only hash and Python syntax check")
    parser.add_argument("--output-parent", type=Path, default=Path(tempfile.gettempdir()))
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        if args.check:
            check_sources(args.vllm_package, args.ascend_package, manifest)
            print(json.dumps({"source_check": "passed", "base_commits": BASES}))
        else:
            print(stage(args.vllm_package, args.ascend_package, manifest, args.output_parent))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Ascend staging failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
