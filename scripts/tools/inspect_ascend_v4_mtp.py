"""Read-only V4 deployment inventory. Uses stdlib; never imports NPU packages."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def inspect(vllm_package=None, ascend_package=None, model=None):
    lock = json.loads((ROOT / "patches/ascend/deepseek-v4-mtp/base_files.json").read_text())
    packages, versions, revisions = {}, {}, {}
    for name, supplied, distribution in (("vllm", vllm_package, "vllm"),
                                       ("vllm_ascend", ascend_package, "vllm-ascend")):
        if supplied is None:
            spec = importlib.util.find_spec(name)
            supplied = Path(spec.origin).parent if spec and spec.origin else None
        packages[name] = str(Path(supplied).resolve()) if supplied else None
        try:
            versions[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        revisions[name] = None
        if supplied:
            result = subprocess.run(["git", "-C", str(supplied), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                revisions[name] = result.stdout.strip()
    differences = []
    for name, expected in lock["sha256"].items():
        package, relative = name.split("/", 1)
        path = Path(packages[package]) / relative if packages[package] else None
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None
        if actual != expected:
            differences.append({"file": name, "expected_sha256": expected, "actual_sha256": actual})
    architecture = None
    if model:
        cfg = json.loads((Path(model) / "config.json").read_text())
        architecture = {key: cfg.get(key) for key in ("architectures", "model_type", "num_hidden_layers", "num_hash_layers")}
        if cfg.get("num_nextn_predict_layers", 1) != 1:
            differences.append({"model": "requires a single MTP layer"})
        if cfg.get("architectures") != ["DeepseekV4ForCausalLM"]:
            differences.append({"model": "requires text-only DeepseekV4ForCausalLM"})
    return {"python": sys.executable, "packages": packages, "versions": versions,
            "git_revisions": revisions, "reference_commits": lock["base_commits"],
            "model": architecture, "source_check_passed": not differences,
            "differences": differences, "npu_verified": False,
            "scope": "locked Python files only; does not verify binary/driver compatibility"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-package", type=Path)
    parser.add_argument("--ascend-package", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    try:
        report = inspect(args.vllm_package, args.ascend_package, args.model)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["source_check_passed"] else 2
    except (OSError, ValueError) as exc:
        print(json.dumps({"source_check_passed": False, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

