"""Run an existing Decode command with an isolated, checked capture overlay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def prepare_environment(overlay, configuration):
    overlay, configuration = Path(overlay).resolve(), Path(configuration).resolve()
    manifest = json.loads((overlay / "overlay.json").read_text())
    if manifest["profile"] != "deepseek-v4-023post1-async-graph" or not manifest.get("debug_entry_ready"):
        raise ValueError("wrong overlay profile")
    if not manifest.get("plugin_sha256"):
        raise ValueError("overlay has no plugin fingerprints; create a fresh overlay")
    for name, expected in manifest["plugin_sha256"].items():
        if hashlib.sha256((ROOT/"src"/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"plugin changed after staging; create a fresh overlay: {name}")
    for name, expected in manifest["sha256"].items():
        package, relative = name.split("/", 1)
        path = overlay / name if package == "vllm_ascend" else Path(manifest["packages"][package]) / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"overlay/source integrity mismatch: {name}")
    sys.path.insert(0, str(ROOT / "src"))
    from moe_trace.adapters.ascend_023_debug import ENV, config
    environment = os.environ.copy()
    environment[ENV] = str(configuration)
    prior = os.environ.get(ENV)
    try:
        os.environ[ENV] = str(configuration)
        cfg = config()
    finally:
        if prior is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = prior
    # Refuse reuse, including an interrupted or failed run. Never remove user output.
    run_node = Path(cfg["output_root"]) / cfg["run_id"] / cfg["node_id"]
    if run_node.exists():
        raise ValueError(f"output already exists; choose a new run_id or archive it: {run_node}")
    run_node.parent.mkdir(parents=True, exist_ok=True)
    # Test host-side output permissions with the same uid as the service launcher.
    import tempfile
    with tempfile.TemporaryFile(dir=run_node.parent):
        pass
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(overlay), str(ROOT / "src"), str(Path(manifest["packages"]["vllm"]).parent),
        environment.get("PYTHONPATH"))))
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="validate only; do not launch")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- bash original_decode.sh args...")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not args.check and not command:
        parser.error("provide the original Decode command after --")
    try:
        environment = prepare_environment(args.overlay, args.config)
        # Validate path precedence without importing vendor code or loading NPU drivers.
        probe = subprocess.run([sys.executable, "-c",
            "import importlib.util,json;print(json.dumps({n:importlib.util.find_spec(n).origin for n in ('vllm','vllm_ascend','moe_trace')}))"],
            env=environment, capture_output=True, text=True, check=True)
        origins = json.loads(probe.stdout)
        if not Path(origins["vllm_ascend"]).resolve().is_relative_to(args.overlay.resolve()):
            raise ValueError("service Python does not resolve the overlay")
        if args.check:
            print(json.dumps({"launch_check": "passed", "origins": origins, "npu_verified": False}))
            return 0
        print("moe-trace: debug capture enabled; original command unchanged; device synchronization reduces throughput.", flush=True)
        os.execvpe(command[0], command, environment)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
