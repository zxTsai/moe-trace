"""Execute pinned patched source on CPU and persist trace/JSON/CSV evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascend-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="qwen36-cpu-", dir=args.output_root.resolve()))
    env = {**os.environ, "MOE_QWEN_ASCEND_SOURCE": str(args.ascend_source.resolve())}
    result = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=", "-q",
        "tests/test_qwen36_capture.py::test_cpu_actual_patched_forward_prepare_and_w8a8",
        "--basetemp", str(output / "pytest")], cwd=ROOT, env=env)
    if result.returncode:
        return result.returncode
    runs = [p for p in (output / "pytest").glob("test_cpu_actual*/cpu") if not p.parent.is_symlink()]
    if len(runs) != 1:
        raise RuntimeError("expected exactly one CPU trace run")
    result = subprocess.run([sys.executable, "-m", "moe_trace.distributed.qwen_cli", str(runs[0]),
        "--output", str(output / "details.json"), "--counts", str(output / "expert-counts.csv")], cwd=ROOT)
    if result.returncode:
        return result.returncode
    report = {"validation_level": "cpu_patched_source_hooks", "npu_verified": False,
        "full_service_verified": False, "real_model_loaded": False,
        "actual_methods": ["patched forward block", "All2All.prepare", "W8A8 MoE.apply"],
        "mock_dependencies": ["model forward callback", "router implementation", "NPU expert kernel", "distributed groups"],
        "independent_expected_comparison": "passed", "logical_routes": 8, "expert_choices": 16,
        "run_dir": str(runs[0]), "details": str(output / "details.json")}
    (output / "cpu-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
