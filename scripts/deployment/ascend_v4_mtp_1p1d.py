"""Two-node 1P1D debug launcher. Dry-run by default; no model/dependency download."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]


def commands(role, model, address, base_port=7100, rpc_port=12321):
    dp, tp = (4, 4) if role == "P" else (16, 1)
    kv = {"kv_connector": "MooncakeHybridConnector", "kv_role": "kv_producer" if role == "P" else "kv_consumer",
          "kv_port": "30000" if role == "P" else "30100", "engine_id": "0" if role == "P" else "1",
          "kv_connector_extra_config": {"prefill": {"dp_size": 4, "tp_size": 4}, "decode": {"dp_size": 16, "tp_size": 1}}}
    additional = ({"enable_cpu_binding": True, "enable_shared_expert_dp": True, "enable_dsa_cp": True} if role == "P" else {
                       "ascend_compilation_config": {"enable_npugraph_ex": False, "enable_static_kernel": False},
                       "enable_cpu_binding": True, "enable_fused_mc2": 0, "enable_dsa_cp": False,
                       "enable_shared_expert_dp": False,
                       "enable_flashcomm1": False, "multistream_overlap_shared_expert": False,
                       "multistream_overlap_gate": False, "recompute_scheduler_enable": True})
    result = []
    for rank in range(dp):
        argv = ["vllm", "serve", str(model), "--host", "0.0.0.0", "--port", str(base_port + rank),
                "--data-parallel-size", str(dp), "--data-parallel-rank", str(rank),
                "--data-parallel-address", address, "--data-parallel-rpc-port", str(rpc_port),
                "--tensor-parallel-size", str(tp), "--enable-expert-parallel", "--seed", "1024",
                "--served-model-name", "dsv4", "--max-model-len", "1048576",
                "--max-num-batched-tokens", "8192" if role == "P" else "120",
                "--max-num-seqs", "16" if role == "P" else "60", "--block-size", "128",
                "--trust-remote-code", "--tokenizer-mode", "deepseek_v4", "--reasoning-parser", "deepseek_v4",
                "--tool-call-parser", "deepseek_v4", "--enable-auto-tool-choice",
                "--model-loader-extra-config", '{"enable_multithread_load":true,"num_threads":128}',
                "--safetensors-load-strategy", "prefetch", "--gpu-memory-utilization", "0.9",
                "--quantization", "ascend", "--enforce-eager", "--no-async-scheduling",
                "--speculative-config", '{"method":"mtp","num_speculative_tokens":1,"enforce_eager":true}',
                "--no-enable-prefix-caching", "--no-disable-hybrid-kv-cache-manager",
                "--compilation-config", '{"cudagraph_mode":"NONE"}',
                "--additional-config", json.dumps(additional), "--kv-transfer-config", json.dumps(kv)]
        result.append({"dp_rank": rank, "devices": ",".join(str(i) for i in range(rank * tp, (rank + 1) * tp)),
                       "command": argv})
    return result


def verify_overlay(overlay):
    expected = json.loads((ROOT / "patches/ascend/deepseek-v4-mtp/debug-overlay.json").read_text())
    actual = json.loads((overlay / "debug-overlay.json").read_text())
    if actual != expected:
        raise ValueError("overlay differs from pinned debug profile")
    for path, digest in expected["patched_sha256"].items():
        if hashlib.sha256((overlay / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"overlay changed: {path}")
    lock = json.loads((ROOT / "patches/ascend/deepseek-v4-mtp/base_files.json").read_text())
    for path, digest in lock["sha256"].items():
        if path.startswith("vllm_ascend/") and path not in expected["patched_sha256"]:
            if hashlib.sha256((overlay / path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"overlay base changed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("P", "D"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--nic", required=True)
    parser.add_argument("--base-port", type=int, default=7100)
    parser.add_argument("--rpc-port", type=int, default=12321)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/v4-mtp"))
    parser.add_argument("--node-id", default="D0")
    parser.add_argument("--save-token-ids", action="store_true")
    parser.add_argument("--save-expert-weights", action="store_true")
    parser.add_argument("--trace", action="store_true", help="D only; baseline has no tracing")
    parser.add_argument("--execute", action="store_true", help="Actually start own server subprocesses")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.node_id) or args.node_id in (".", ".."):
        parser.error("node-id must be a safe identifier")
    if args.trace and (args.role != "D" or args.overlay is None):
        parser.error("--trace requires role D and --overlay")
    if not 1 <= args.base_port <= 65520 or not 1 <= args.rpc_port <= 65535:
        parser.error("invalid ports")
    if args.base_port <= args.rpc_port < args.base_port + (4 if args.role == "P" else 16):
        parser.error("RPC port overlaps API ports")
    tasks = commands(args.role, args.model.resolve(), args.local_ip, args.base_port, args.rpc_port)
    if not args.execute:
        print(json.dumps({"profile": "deepseek-v4-mtp-debug", "execute": False, "trace": args.trace,
                          "npu_verified": False, "tasks": tasks}, indent=2))
        return 0
    if not args.model.is_dir():
        parser.error("model directory does not exist; download is not performed")
    model_config = json.loads((args.model / "config.json").read_text())
    if model_config.get("architectures") != ["DeepseekV4ForCausalLM"]:
        parser.error("requires text-only DeepseekV4ForCausalLM; model directory name is not proof")
    env = os.environ.copy()
    defaults = {
        "TASK_QUEUE_ENABLE": "1", "VLLM_RPC_TIMEOUT": "3600000",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "30000", "HCCL_EXEC_TIMEOUT": "204",
        "HCCL_CONNECT_TIMEOUT": "120" if args.role == "P" else "1200",
        "OMP_PROC_BIND": "false", "OMP_NUM_THREADS": "10",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    jemalloc = "/usr/lib64/libjemalloc.so.2"
    if Path(jemalloc).is_file() and jemalloc not in env.get("LD_PRELOAD", "").split(":"):
        env["LD_PRELOAD"] = jemalloc + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    # A baseline must not inherit another tracing overlay/config from the caller.
    if not args.trace and (env.get("MOE_TRACE_V4_MTP_CONFIG") or env.get("MOE_TRACE_V41_SOURCE_HOOKS")
                           or env.get("MOE_TRACE_CONFIG") or env.get("MOE_TRACE_QWEN36_CONFIG")
                           or env.get("MOE_TRACE_V41_DEBUG_CONFIG") or env.get("MOE_TRACE_V4_DEBUG_CONFIG")):
        parser.error("baseline must run in a clean shell without tracing configuration")
    env.update({"VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_ENGINE_READY_TIMEOUT_S": "36000",
                "HCCL_IF_IP": args.local_ip, "GLOO_SOCKET_IFNAME": args.nic,
                "TP_SOCKET_IFNAME": args.nic, "HCCL_SOCKET_IFNAME": args.nic,
                "HCCL_BUFFSIZE": "2560" if args.role == "P" else "1024"})
    env["HCCL_OP_EXPANSION_MODE"] = "AIV"
    env["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "1" if args.role == "P" else "0"
    if args.trace:
        overlay = args.overlay.resolve()
        verify_overlay(overlay)
        if any(env.get(key) for key in ("MOE_TRACE_V41_SOURCE_HOOKS", "MOE_TRACE_CONFIG",
                                       "MOE_TRACE_V41_DEBUG_CONFIG", "MOE_TRACE_QWEN36_CONFIG", "MOE_TRACE_V4_DEBUG_CONFIG")):
            parser.error("do not combine debug hooks with other tracing tools")
        # Import/path/hash preflight uses the same interpreter/environment as the service CLI.
        env["PYTHONPATH"] = str(overlay) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([sys.executable, "-c",
                        "import importlib.util; from pathlib import Path; "
                        "import moe_trace.adapters.ascend_v4_mtp; "
                        "p=Path(importlib.util.find_spec('vllm_ascend').origin).resolve(); "
                        f"assert p.is_relative_to(Path({str(overlay)!r})), p"], env=env, check=True)
        # The unchanged vLLM package must match the locked metadata/rejection source files.
        vllm_package = subprocess.check_output([sys.executable, "-c",
            "import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec('vllm').origin).parent)"],
            env=env, text=True).strip()
        lock = json.loads((ROOT / "patches/ascend/deepseek-v4-mtp/base_files.json").read_text())
        for path, digest in lock["sha256"].items():
            if path.startswith("vllm/"):
                if hashlib.sha256((Path(vllm_package) / path.split("/", 1)[1]).read_bytes()).hexdigest() != digest:
                    parser.error(f"installed vLLM source mismatch: {path}")
        run_id = str(uuid.uuid4())
        directory = args.output_root.resolve() / run_id
        directory.mkdir(parents=True, exist_ok=False)
        config = {"enabled": True, "output_root": str(args.output_root.resolve()), "run_id": run_id,
                  "node_id": args.node_id, "save_token_ids": args.save_token_ids,
                  "save_expert_weights": args.save_expert_weights}
        # Use standard library; no environment installation or user config overwrite.
        (directory / "debug-config.json").write_text(json.dumps(config, indent=2) + "\n")
        (directory / "deployment.json").write_text(json.dumps({
            "profile": "deepseek-v4-mtp-debug", "run_id": run_id, "node_id": args.node_id,
            "expected_workers": [f"dp-{d}-tp-{t}" for d in range(16) for t in range(1)],
            "npu_verified": False, "tasks": tasks}, indent=2) + "\n")
        env["MOE_TRACE_V4_MTP_CONFIG"] = str(directory / "debug-config.json")
        print(f"Trace run: {directory}", flush=True)
    processes = []
    stopping = False
    def stop_children(signum, frame):
        nonlocal stopping
        stopping = True
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)  # Only our own process groups.
                except ProcessLookupError:
                    pass
    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)
    try:
        for task in tasks:
            child_env = {**env, "ASCEND_RT_VISIBLE_DEVICES": task["devices"]}
            argv = [sys.executable, "-m", "vllm.entrypoints.cli.main", *task["command"][1:]]
            processes.append(subprocess.Popen(argv, env=child_env, start_new_session=True))
        while not stopping:
            active = [p for p in processes if p.poll() is None]
            if not active:
                break
            try:
                active[0].wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            if any(process.poll() not in (None, 0) for process in processes):
                stop_children(None, None)
                break
    finally:
        stop_children(None, None)
        deadline = time.monotonic() + 30
        for process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        process.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        continue
    return 0 if all(p.returncode == 0 for p in processes) else 1


if __name__ == "__main__":
    raise SystemExit(main())


