"""Create an isolated patched Python package; preserve the installed environment."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path, help="installed vllm package directory")
    args = parser.parse_args()
    source = args.package.resolve()
    patches = Path(__file__).resolve().parents[1] / "patches/vllm/0.29.0"
    manifest = json.loads((patches / "base_files.json").read_text())
    for relative, expected in manifest["sha256"].items():
        actual = hashlib.sha256((source / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"base source differs: {relative}; refusing to patch")
    root = Path(tempfile.mkdtemp(prefix="moe-trace-vllm-029-"))

    def copy_file(src, dst):
        if str(src).endswith(".so"):
            Path(dst).symlink_to(Path(src).resolve())
            return dst
        return shutil.copy2(src, dst)

    target = root / "vllm"
    shutil.copytree(source, target, copy_function=copy_file,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patch = patches / "0001-trace-hooks.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=target, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=target, check=True)
    print(root)


if __name__ == "__main__":
    main()
