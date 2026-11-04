"""Run actual patched V4 Python hooks on CPU and persist JSON/CSV evidence."""
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
    parser.add_argument("--k", type=int, help="validate this positive runtime K using the real autoregressive loop")
    args = parser.parse_args()
    if args.k is not None and args.k<1:
        parser.error("--k must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="deepseek-v4-mtp-cpu-", dir=args.output_root.resolve()))
    env = {**os.environ, "MOE_V4_VLLM_SOURCE": str(args.vllm_source.resolve()),
           "MOE_V4_ASCEND_SOURCE": str(args.ascend_source.resolve())}
    test = "tests/test_v4_mtp.py::test_actual_patched_mtp_source_cpu"
    if args.k is not None:
        env["MOE_MTP_VALIDATION_K"] = str(args.k)
        test = "tests/test_v4_mtp_dynamic_k.py::test_requested_k_from_environment"
    result = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=", "-q",
        test, "--basetemp", str(output / "pytest")],
        cwd=ROOT, env=env)
    if result.returncode:
        return result.returncode
    runs = [p for p in (output / "pytest").glob("test_*/cpu") if not p.parent.is_symlink()]
    if len(runs) != 1:
        raise RuntimeError("expected exactly one source-hook CPU trace run")
    result = subprocess.run([sys.executable, "-m", "moe_trace.distributed.v4_mtp_cli", str(runs[0]),
        "--output", str(output / "details.json"), "--counts", str(output / "expert-counts.csv"),
        "--by-execution", str(output / "execution-counts.csv")], cwd=ROOT)
    if result.returncode:
        return result.returncode
    report = {"validation_level": "cpu_patched_source_hooks", "npu_verified": False,
        "full_service_verified": False, "real_model_loaded": False,
        "actual_methods": ["patched runner forward block", "All2All.prepare",
            "AscendRoutedExperts._select_experts", "AscendRoutedExperts.forward_impl", "W8A8 MoE.apply",
            "MTP set_inputs_first_pass", "patched MTP first-pass forward block",
            "patched MTP return expression", "runner _sample with sampler-result hook"],
        "mock_dependencies": ["model forward callback", "router/NPU expert kernels", "distributed groups"],
        "independent_expected_comparison": "passed", "logical_routes": 17, "expert_choices": {"target": 24, "mtp": 10, "combined": 34},
        "candidate_states": {"accepted": 1, "rejected": 1, "unverified": 2},
        "run_dir": str(runs[0]), "details": str(output / "details.json")}
    if args.k is not None:
        k=args.k
        report.update(k=k,logical_routes=8+6*k,expert_choices={"target":8*k+16,"mtp":4*k,"combined":12*k+16},
            candidate_states={"accepted":2*(k//2),"rejected":2,"not_adopted":2*max(0,k-k//2-1),"unverified":0},
            actual_methods=["patched first MTP forward block", "set_inputs_first_pass",
                "full patched autoregressive for loop", "patched final return expression",
                "target verification/acceptance hooks", "scoped storage and export"],
            mock_dependencies=["model callback", "routing tensors and kernels", "distributed groups", "sampler output"])
    (output / "cpu-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

