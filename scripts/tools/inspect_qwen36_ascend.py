"""Read-only Qwen3.6 environment inventory; does not import NPU packages or patch them."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def package_info(name):
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        spec = None
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    path = Path(spec.origin).resolve().parent if spec and spec.origin else None
    return {"version": version, "package_path": str(path) if path else None}


def git_head(package):
    # Only inspect an immediate source checkout, not an unrelated enclosing repository.
    root = package.parent
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def inspect_sources(package):
    lock = json.loads((ROOT / "patches/ascend/qwen36-023post1/source-lock.json").read_text())
    files = {}
    for name, expected in lock["sha256"].items():
        path = Path(package) / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        files[name] = {"sha256": actual, "matches_reference": actual == expected}
    return {"reference_commit": lock["ascend_commit"], "git_head": git_head(Path(package)),
            "all_reference_files_match": all(v["matches_reference"] for v in files.values()),
            "files": files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Optional local model directory; reads config.json only")
    parser.add_argument("--ascend-package", type=Path, help="Source inventory override; not an import-path check")
    parser.add_argument("--output", type=Path, help="Save inventory to a new file, never overwrite")
    args = parser.parse_args(argv)
    try:
        packages = {name: package_info(name) for name in ("vllm", "vllm_ascend", "torch", "torch_npu")}
        selected = args.ascend_package or packages["vllm_ascend"]["package_path"]
        report = {"profile": "qwen36-single-node-dp8-tp2-inventory", "python": sys.version,
                  "executable": sys.executable, "packages": packages,
                  "source_path_override": args.ascend_package is not None,
                  "runtime_supported": False, "npu_verified": False,
                  "startup_command_reviewed": False,
                  "limitations": ["Source matches do not validate NPU binaries or runtime configuration",
                                  "Experimental collector has CPU hook tests; this inventory does not validate runtime support"]}
        report["ascend_sources"] = inspect_sources(selected) if selected else None
        vllm = packages["vllm"]["package_path"]
        report["vllm_git_head"] = git_head(Path(vllm)) if vllm else None
        if args.model:
            cfg = json.loads((args.model / "config.json").read_text())
            text = cfg.get("text_config", cfg)
            report["model"] = {"architectures": cfg.get("architectures"),
                               "model_type": cfg.get("model_type"),
                               "text_model_type": text.get("model_type"),
                               "num_experts": text.get("num_experts"),
                               "num_experts_per_tok": text.get("num_experts_per_tok"),
                               "num_hidden_layers": text.get("num_hidden_layers"),
                               "quantization_config": cfg.get("quantization_config")}
        encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            # Exclusive creation prevents replacing a user's existing report.
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        print(encoded, end="")
    except (OSError, ValueError, TypeError) as exc:
        print(f"Inventory failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
