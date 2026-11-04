"""Read installed source fingerprints without Git, vendor imports or service changes."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "patches/ascend/deepseek-v4-023post1-async-graph/reference-files.json"


def locate(package, explicit, workspace):
    if explicit is not None:
        return Path(explicit).resolve(), "explicit"
    spec = importlib.util.find_spec(package)
    if spec and spec.origin:
        return Path(spec.origin).resolve().parent, "installed_import_spec"
    directory = "vllm" if package == "vllm" else "vllm-ascend"
    return (Path(workspace) / directory / package).resolve(), "default_workspace"


def inspect_sources(vllm_package=None, ascend_package=None, workspace=Path("/vllm-workspace")):
    reference = json.loads(REFERENCE.read_text())
    roots, discovery, versions, actual, differences = {}, {}, {}, {}, []
    for package, explicit, distribution in (("vllm", vllm_package, "vllm"),
                                           ("vllm_ascend", ascend_package, "vllm-ascend")):
        roots[package], discovery[package] = locate(package, explicit, workspace)
        try:
            versions[package] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    for name, expected in reference["sha256"].items():
        package, relative = name.split("/", 1)
        path = roots[package] / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        actual[name] = digest
        if digest != expected:
            differences.append({"file": name, "expected_sha256": expected, "actual_sha256": digest})
    return {"profile": reference["profile"], "target_image_tag": reference["image_tag"],
            "python": sys.executable, "versions": versions,
            "packages": {k: str(v) for k, v in roots.items()}, "discovery": discovery,
            "reference_source_match": not differences, "sha256": actual, "differences": differences,
            "adapter_ready": False, "debug_entry_ready": reference.get("debug_entry_ready", False),
            "npu_verified": False,
            "note": "Read-only source inventory. Debug overlay is available; source match is not A3 acceptance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/vllm-workspace"))
    parser.add_argument("--vllm-package", type=Path)
    parser.add_argument("--ascend-package", type=Path)
    parser.add_argument("--output", type=Path, help="optional new report file; never overwrite")
    args = parser.parse_args()
    try:
        report = inspect_sources(args.vllm_package, args.ascend_package, args.source_root)
        data = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(data)
        print(data, end="")
        return 0 if report["reference_source_match"] else 2
    except (OSError, ValueError, ImportError) as exc:
        print(json.dumps({"reference_source_match": False, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
