"""Mechanically regenerate Ascend hooks from clean v0.23.0 package sources."""
import argparse
import runpy
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-package", type=Path, required=True)
    parser.add_argument("--ascend-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    implementation = runpy.run_path(str(Path(__file__).resolve().parents[2]
                                       / "src/moe_trace/setup/ascend.py"))
    implementation["generate"](args.vllm_package, args.ascend_package, args.output)
