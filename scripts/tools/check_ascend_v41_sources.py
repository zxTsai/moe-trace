"""Read-only source inventory check; does not install or patch a service."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def check_sources(vllm_root, ascend_root, lock):
    if lock.get("validation_level") != "source_inventory_only":
        raise ValueError("unsupported source inventory")
    roots = {"vllm": Path(vllm_root), "vllm_ascend": Path(ascend_root)}
    for name, root in roots.items():
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.PIPE).strip()
        if actual != lock["base_commits"][name]:
            raise ValueError(f"wrong source commit: {name}")
    for path, expected in lock["sha256"].items():
        name = path.split("/", 1)[0]
        data = (roots[name] / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"wrong source hash: {path}")
        ast.parse(data.decode("utf-8"), filename=path)
    return {"source_inventory": "passed", "base_commits": lock["base_commits"],
            "files": len(lock["sha256"]), "npu_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True, help="Git repository, not package directory")
    parser.add_argument("--ascend-root", type=Path, required=True, help="Git repository, not package directory")
    parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().parents[2]
                        / "docs/design/ascend-v41-source-lock.json")
    args = parser.parse_args()
    try:
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        print(json.dumps(check_sources(args.vllm_root, args.ascend_root, lock), indent=2))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Source inventory failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
