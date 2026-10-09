"""Single-node Qwen debug capture launcher. Dry-run by default, A3 unverified."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]


def verify_overlay(overlay):
    folder = ROOT / "patches/ascend/qwen36-023post1"
    expected = json.loads((folder / "overlay.json").read_text())
    if json.loads((overlay / "overlay.json").read_text()) != expected:
        raise ValueError("overlay metadata mismatch")
    lock = json.loads((folder / "source-lock.json").read_text())
    digests = {**lock["sha256"], **expected["patched_sha256"]}
    for path, digest in digests.items():
        if hashlib.sha256((overlay / "vllm_ascend" / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"overlay source changed: {path}")


def command(model):
    additional = {"enable_flashcomm1": False, "enable_sp_by_pass": False, "enable_fused_mc2": 0,
                  "enable_dsa_cp": False, "enable_shared_expert_dp": False,
                  "multistream_overlap_gate": False, "multistream_overlap_shared_expert": False}
    return [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model),
        "--host", "127.0.0.1", "--port", "8000", "--served-model-name", "qwen3.6",
        "--data-parallel-size", "8", "--data-parallel-size-local", "8",
        "--tensor-parallel-size", "2", "--enable-expert-parallel",
        "--quantization", "ascend", "--enforce-eager", "--no-async-scheduling",
        "--no-enable-prefix-caching", "--max-model-len", "4096", "--max-num-seqs", "8",
        "--max-num-batched-tokens", "512", "--gpu-memory-utilization", "0.8",
        "--additional-config", json.dumps(additional)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--node-id", default="A3")
    parser.add_argument("--save-token-ids", action="store_true")
    parser.add_argument("--save-expert-weights", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    import re
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.node_id) or args.node_id in {".", ".."}:
        parser.error("unsafe node ID")
    if not 1 <= args.port <= 65535:
        parser.error("invalid port")
    argv = command(args.model.resolve())
    argv[argv.index("--host") + 1] = args.host
    argv[argv.index("--port") + 1] = str(args.port)
    if not args.execute:
        print(json.dumps({"execute": False, "npu_verified": False, "command": argv,
                          "output_root": str(args.output_root.resolve())}, indent=2))
        return 0
    if not args.model.is_dir():
        parser.error("model directory missing")
    verify_overlay(args.overlay.resolve())
    if not importlib.metadata.version("vllm").startswith("0.23.0"):
        parser.error("requires inspected vLLM 0.23.0 environment")
    if importlib.metadata.version("vllm_ascend") != "0.23.0.post1":
        parser.error("requires Ascend 0.23.0.post1")
    env = os.environ.copy()
    if any(env.get(k) for k in ("MOE_TRACE_CONFIG", "MOE_TRACE_V41_DEBUG_CONFIG", "MOE_TRACE_V41_SOURCE_HOOKS", "MOE_TRACE_QWEN36_CONFIG")):
        parser.error("start in a clean shell without another capture config")
    env["PYTHONPATH"] = str(args.overlay.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    subprocess.run([sys.executable, "-c", "import importlib.util; from pathlib import Path; "
        "import moe_trace.adapters.ascend_qwen36; "
        "p=Path(importlib.util.find_spec('vllm_ascend').origin).resolve(); "
        f"assert p.is_relative_to(Path({str(args.overlay.resolve())!r})), p"], env=env, check=True)
    cfg = dict(enabled=True, output_root=str(args.output_root.resolve()), run_id=str(uuid.uuid4()),
               node_id=args.node_id, save_token_ids=args.save_token_ids,
               save_expert_weights=args.save_expert_weights)
    run = Path(cfg["output_root"]) / cfg["run_id"]
    run.mkdir(parents=True, exist_ok=False)
    config = run / "capture-config.json"
    config.write_text(json.dumps(cfg, indent=2) + "\n")
    (run / "launch.json").write_text(json.dumps({"profile": "qwen36-single-node-debug",
        "command": argv, "npu_verified": False, "requests_terminal_verified": False}, indent=2) + "\n")
    env["MOE_TRACE_QWEN36_CONFIG"] = str(config)
    print(f"Trace run: {run}", flush=True)
    # Replace launcher with the service; terminal signals go directly to vLLM.
    os.execvpe(sys.executable, argv, env)


if __name__ == "__main__":
    raise SystemExit(main())
