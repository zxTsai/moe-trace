"""Run current reference-source CPU checks and save reproducible evidence."""
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
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--ascend-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="ascend-023-source-cpu-", dir=args.output_root.resolve()))
    environment = {**os.environ, "MOE_VLLM_023_SOURCE": str(args.vllm_source.resolve()),
                   "MOE_ASCEND_023_SOURCE": str(args.ascend_source.resolve())}
    command = [sys.executable, "-m", "pytest", "-o", "addopts=", "-q",
        "tests/test_ascend_023_source.py", "tests/test_ascend_023_debug.py",
        "tests/test_ascend_graph_bridge.py", "tests/test_graph_routes.py", "tests/test_ascend_async_inventory.py",
        "tests/test_ascend_load_summary.py",
        "--basetemp", str(output / "pytest")]
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    (output/"pytest.log").write_text(result.stdout+result.stderr)
    report = dict(validation_level="cpu_reference_source_with_hardware_stubs",
        modes=["trace","summary"],
        passed=result.returncode == 0, npu_verified=False, full_service_verified=False,
        real_model_loaded=False, vllm_source=str(args.vllm_source.resolve()),
        ascend_source=str(args.ascend_source.resolve()),
        actual_methods=["runner forward block", "W8A8 dynamic apply", "runner _sample",
                        "MTP set_inputs_first_pass", "MTP _run_merged_draft K=1/2/3", "ACLGraphWrapper.__call__"],
        mocks=["router and expert kernel", "model and sampling output", "NPU graph/streams", "distributed groups"],
        test_log=str(output/"pytest.log"))
    (output/"cpu-validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(result.stdout,end="")
    print(result.stderr,end="",file=sys.stderr)
    print(output/"cpu-validation.json")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
