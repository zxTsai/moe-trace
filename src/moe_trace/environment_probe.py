"""Standalone stdlib probe, executed by the environment being inspected."""
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path


def inspect_environment(gpu=False):
    versions = {}
    for name in ("vllm", "torch", "transformers", "numpy", "PyYAML", "setuptools", "moe-trace"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    spec = importlib.util.find_spec("vllm")
    report = dict(python=sys.executable, prefix=sys.prefix,
                  python_version=platform.python_version(), platform=sys.platform,
                  machine=platform.machine(), packages=versions,
                  vllm_package=str(Path(spec.origin).parent) if spec and spec.origin else None)
    if gpu:
        result = {"available": False, "verified": False}
        try:
            import torch
            result["available"] = torch.cuda.is_available()
            if not result["available"]:
                raise RuntimeError("CUDA is unavailable to this process")
            result["device"] = torch.cuda.get_device_name(0)
            result["cuda_runtime"] = torch.version.cuda
            value = (torch.ones(2, device="cuda") * 3).sum().item()
            if value != 6:
                raise RuntimeError("CUDA arithmetic check failed")
            version = versions["vllm"] or ""
            if version == "0.29.0":
                import vllm._C_stable_libtorch
                import vllm._moe_C_stable_libtorch
            elif version.split("+")[0] == "0.10.1":
                import vllm._C
                import vllm._moe_C
            else:
                raise RuntimeError("unsupported vLLM version for native verification")
            result["verified"] = True
        except Exception as exc:
            result["error"] = str(exc)
        report["gpu"] = result
    return report


if __name__ == "__main__":
    # Imports may print warnings; a tagged final line keeps parsing unambiguous.
    print("MOE_TRACE_ENV_JSON=" + json.dumps(inspect_environment("--gpu" in sys.argv)))
