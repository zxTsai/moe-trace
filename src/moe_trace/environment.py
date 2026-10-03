"""Opt-in environment preparation. Stdlib only; never starts a model service."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROFILES = {
    "0.10.1": {"python": "3.11", "torch": "2.7.1", "transformers": "4.55.2"},
    "0.29.0": {"python": "3.12", "torch": "2.13.0", "transformers": "5.18.0"},
}
PATCHES = {
    "0.10.1": ["0001-moe-trace-hooks.patch", "0002-uniproc-trace-shutdown.patch",
               "0003-request-lifecycle.patch"],
    "0.29.0": ["0001-trace-hooks.patch"],
}


class SetupError(RuntimeError):
    pass


def clean_environment():
    env = os.environ.copy()
    # Detect the real installed package, not an overlay left in the caller's shell.
    for key in ("PYTHONPATH", "PYTHONHOME", "MOE_TRACE_CONFIG", "MOE_TRACE_REFERENCE_DIR"):
        env.pop(key, None)
    return env


def run(command, *, capture=False, cwd=None, environment=None):
    result = subprocess.run(command, cwd=cwd, env=environment or clean_environment(), text=True,
                            capture_output=capture)
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip() if capture else "see command output"
        raise SetupError(f"command failed ({result.returncode}): {shlex.join(command)}\n{detail}")
    return result.stdout if capture else ""


def probe(prefix, gpu=False):
    script = str(Path(__file__).with_name("environment_probe.py"))
    output = run([*prefix, script, *( ["--gpu"] if gpu else [])], capture=True)
    lines = [line.removeprefix("MOE_TRACE_ENV_JSON=") for line in output.splitlines()
             if line.startswith("MOE_TRACE_ENV_JSON=")]
    if len(lines) != 1:
        raise SetupError("target interpreter did not return an unambiguous environment report")
    return json.loads(lines[0])


def check_base(package, version, repository):
    manifest = repository / "patches/vllm" / version / "base_files.json"
    if not manifest.is_file():
        return ["versioned patch manifest is missing from this repository"]
    data = json.loads(manifest.read_text())
    problems = []
    for relative, expected in data["sha256"].items():
        path = package / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f"patch base differs: {relative}")
    return problems


def assess(report, repository):
    packages = report["packages"]
    version = (packages.get("vllm") or "").split("+")[0]
    problems = []
    if version not in PROFILES:
        problems.append("vLLM is absent or its version is unsupported")
    if report["platform"] != "linux" or report["machine"].lower() not in {"x86_64", "amd64"}:
        problems.append("this integration currently targets Linux/WSL2 x86_64")
    if tuple(int(x) for x in report["python_version"].split(".")[:2]) < (3, 11):
        problems.append("moe-trace requires Python >=3.11")
    if version in PROFILES:
        for name in ("torch", "transformers"):
            if (packages.get(name) or "").split("+")[0] != PROFILES[version][name]:
                problems.append(f"{name} differs from validated profile {PROFILES[version][name]}")
        for name, minimum in (("numpy", (1, 26)), ("PyYAML", (6, 0)), ("setuptools", (68, 0))):
            match = re.match(r"(\d+)(?:\.(\d+))?", packages.get(name) or "")
            if not match or (int(match[1]), int(match[2] or 0)) < minimum:
                problems.append(f"{name} is missing or below {minimum[0]}.{minimum[1]}")
        if report.get("vllm_package"):
            problems.extend(check_base(Path(report["vllm_package"]), version, repository))
        else:
            problems.append("installed vLLM package could not be located")
    return dict(version=version or None, compatible=not problems, problems=problems)


def conda_inventory(conda):
    data = json.loads(run([conda, "env", "list", "--json"], capture=True))
    return [Path(path).resolve() for path in data.get("envs", [])]


def validate_new_target(name, prefix, environments):
    if prefix:
        target = Path(prefix).resolve()
        if target.exists() or target in environments:
            raise SetupError("new environment prefix already exists; refusing to modify it")
        return ["--prefix", str(target)]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name or ""):
        raise SetupError("invalid Conda environment name")
    if name.lower() in {"base", "root"} or any(path.name == name for path in environments):
        raise SetupError("environment name is reserved or already exists; choose a new name")
    return ["--name", name]


def build_plan(mode, version, repository, interpreter, *, conda=None, target=None):
    if mode == "existing":
        install = [*interpreter, "-m", "pip", "install", "--no-deps",
                   "--no-build-isolation", "-e", str(repository)]
        return {"mode": mode, "version": version, "interpreter": interpreter,
                "commands": [install], "changes": "installs only moe-trace; dependencies are not upgraded"}
    profile = PROFILES[version]
    interpreter = [conda, "run", "--no-capture-output", *target, "python"]
    requirements = [f"vllm=={version}", f"torch=={profile['torch']}",
                    f"transformers=={profile['transformers']}", "numpy>=1.26,<3",
                    "PyYAML>=6", "setuptools>=68", "wheel"]
    return {"mode": mode, "version": version, "interpreter": interpreter,
            "commands": [
                [conda, "create", "--yes", *target, f"python={profile['python']}", "pip"],
                [*interpreter, "-m", "pip", "install", "--only-binary=:all:", *requirements],
                [*interpreter, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                 "-e", str(repository)]],
            "changes": "creates a new environment; downloads binary dependencies; preserves existing environments"}


def stage_package(package, version, repository, root):
    problems = check_base(package, version, repository)
    if problems:
        raise SetupError("\n".join(problems))
    target = root / "overlay" / "vllm"

    def copy_file(src, dst):
        if str(src).endswith(".so"):
            Path(dst).symlink_to(Path(src).resolve())
            return dst
        return shutil.copy2(src, dst)

    shutil.copytree(package, target, copy_function=copy_file,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    git = shutil.which("git")
    if not git:
        raise SetupError("git is required for conflict-checked patch application")
    # Older patch paths include vllm/, newer ones are package-relative.
    cwd = target.parent if version == "0.10.1" else target
    for name in PATCHES[version]:
        patch = str(repository / "patches/vllm" / version / name)
        run([git, "apply", "--check", patch], capture=True, cwd=cwd)
        run([git, "apply", patch], capture=True, cwd=cwd)
    return target.parent


def write_artifacts(root, repository, report, plan, overlay, output_root, gpu_verified):
    output_root = Path(output_root).resolve()
    config = root / "trace.yaml"
    # JSON quoted strings are valid YAML, including paths with spaces/colons.
    config.write_text("trace:\n  enabled: true\n  output_root: " + json.dumps(str(output_root))
                      + "\n  save_token_ids: true\n  save_expert_weights: false\n"
                      "  buffer_records: 8192\n  shard_records: 4096\nvllm:\n"
                      "  enforce_eager: true\n  enable_prefix_caching: false\n"
                      "  tensor_parallel_size: 1\n  pipeline_parallel_size: 1\n"
                      "  data_parallel_size: 1\n"
                      + ("  async_scheduling: false\n" if plan["version"] == "0.29.0" else ""))
    exports = {"PYTHONPATH": f"{overlay}:{repository / 'src'}", "MOE_TRACE_CONFIG": str(config)}
    if plan["version"] == "0.29.0":
        exports.update(VLLM_USE_V2_MODEL_RUNNER="0", VLLM_USE_FLASHINFER_SAMPLER="0")
    # No conda activation, service launch, or persistent shell configuration changes.
    (root / "activate-trace.sh").write_text(
        "# Source this after activating the target environment.\n"
        + f"if [ \"$(python -c 'import sys; print(sys.executable)')\" != {shlex.quote(report['python'])} ]; then\n"
        + "  echo 'Wrong Python environment: activate the prepared target first.' >&2\n  return 1\nfi\n"
        + "\n".join(f"export {key}={shlex.quote(value)}" for key, value in exports.items()) + "\n")
    result = dict(status="prepared" if gpu_verified else "prepared_gpu_unverified",
                  environment=report, plan=plan, overlay=str(overlay), config=str(config),
                  output_root=str(output_root), gpu_verified=gpu_verified,
                  service_started=False, warning="Conda does not isolate GPU/ports; model inference remains unverified")
    (root / "environment.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def execute_plan(plan, repository, output_parent, trace_output, *, skip_gpu=False):
    if plan["mode"] == "new":
        create = plan["commands"][0]
        flag = "--prefix" if "--prefix" in create else "--name"
        value = create[create.index(flag) + 1]
        validate_new_target(value if flag == "--name" else None,
                            value if flag == "--prefix" else None,
                            conda_inventory(create[0]))
    # Existing mode is rechecked before any installation and never upgrades dependencies.
    if plan["mode"] == "existing":
        problems = assess(probe(plan["interpreter"]), repository)["problems"]
        if problems:
            raise SetupError("existing environment is not compatible:\n" + "\n".join(problems))
        run([*plan["interpreter"], "-m", "pip", "check"], capture=True)
    for command in plan["commands"]:
        print("Executing:", shlex.join(command), flush=True)
        run(command)
    report = probe(plan["interpreter"], gpu=not skip_gpu)
    assessment = assess(report, repository)
    if assessment["version"] != plan["version"] or not assessment["compatible"]:
        raise SetupError("installed environment failed compatibility checks:\n" + "\n".join(assessment["problems"]))
    run([*plan["interpreter"], "-m", "pip", "check"], capture=True)
    if not skip_gpu and not report.get("gpu", {}).get("verified"):
        raise SetupError("GPU/native verification failed: " + str(report.get("gpu")))
    parent = Path(output_parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="prepared-", dir=parent))
    overlay = stage_package(Path(report["vllm_package"]), plan["version"], repository, root)
    verification_env = clean_environment()
    verification_env["PYTHONPATH"] = f"{overlay}:{repository / 'src'}"
    # Import in the actual target interpreter, not the bootstrap environment.
    code = ("import pathlib, vllm, moe_trace; "
            f"assert pathlib.Path(vllm.__file__).resolve().parent == pathlib.Path({str(overlay / 'vllm')!r}).resolve(); "
            f"assert pathlib.Path(moe_trace.__file__).resolve().parent == pathlib.Path({str(repository / 'src/moe_trace')!r}).resolve(); "
            "print('Patched vLLM and moe-trace import paths verified')")
    run([*plan["interpreter"], "-c", code], capture=True, environment=verification_env)
    return root, write_artifacts(root, repository, report, plan, overlay, trace_output, not skip_gpu)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect and explicitly prepare an isolated MoE tracing environment")
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--python", default=sys.executable, help="interpreter to inspect")
    source.add_argument("--conda-env", help="existing Conda environment to inspect")
    parser.add_argument("--conda", default=shutil.which("conda"))
    parser.add_argument("--detect", action="store_true", help="read-only JSON inspection")
    parser.add_argument("--gpu", action="store_true", help="also probe CUDA in detect mode")
    parser.add_argument("--mode", choices=("new", "existing"))
    parser.add_argument("--version", choices=tuple(PROFILES))
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--name", help="new Conda environment name")
    target.add_argument("--prefix", type=Path, help="new Conda environment directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="explicitly approve the printed installation plan")
    parser.add_argument("--skip-gpu-check", action="store_true", help="prepare only; report GPU as unverified")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/environments"))
    parser.add_argument("--trace-output", type=Path, default=Path("outputs/traces"))
    args = parser.parse_args(argv)
    try:
        repository = args.repository.resolve()
        if not (repository / "pyproject.toml").is_file():
            raise SetupError("provide --repository pointing to a complete moe-trace source checkout")
        if args.conda_env and not args.conda:
            raise SetupError("conda executable not found")
        interpreter = ([args.conda, "run", "--no-capture-output", "--name", args.conda_env, "python"]
                       if args.conda_env else [args.python])
        report = probe(interpreter, gpu=args.gpu and args.detect)
        assessment = assess(report, repository)
        if args.detect:
            print(json.dumps({**report, "assessment": assessment}, indent=2))
            return 0
        print(json.dumps({"detected": report, "assessment": assessment}, indent=2))
        interactive = sys.stdin.isatty() and not (args.yes or args.dry_run)
        mode = args.mode
        if not mode:
            if interactive:
                choice = input("Create a NEW dedicated Conda environment [1, recommended], or use EXISTING [2]? (1): ").strip()
                if choice not in {"", "1", "2"}:
                    raise SetupError("invalid environment choice")
                mode = "existing" if choice == "2" else "new"
            elif args.yes:
                raise SetupError("noninteractive approval requires explicit --mode new or existing")
            else:
                mode = "new"
        version = args.version or (assessment["version"] if assessment["version"] in PROFILES else "0.29.0")
        if not args.version and interactive and (mode == "new"):
            version = input(f"vLLM version [0.10.1/0.29.0] ({version}): ").strip() or version
            if version not in PROFILES:
                raise SetupError("unsupported version")
        if mode == "existing":
            if args.name or args.prefix:
                raise SetupError("--name/--prefix are only for NEW environments")
            if not assessment["compatible"] or assessment["version"] != version:
                raise SetupError("existing environment does not meet the requested profile; choose --mode new")
            plan = build_plan(mode, version, repository, interpreter)
        else:
            if not args.conda:
                raise SetupError("Conda is required to create an isolated environment; install it separately")
            if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
                raise SetupError("new collection environments currently require Linux/WSL2 x86_64")
            name = args.name or f"moe-trace-vllm-{version.replace('.', '-')}"
            if interactive and not args.name and not args.prefix:
                name = input(f"New environment name ({name}): ").strip() or name
            target_args = validate_new_target(name, args.prefix, conda_inventory(args.conda))
            plan = build_plan(mode, version, repository, interpreter, conda=args.conda, target=target_args)
        print("\nInstallation plan (no services will be started):")
        print(json.dumps(plan, indent=2))
        print("Conda isolates dependencies, not GPU resources or production services.")
        if args.dry_run:
            return 0
        if not args.yes:
            if not interactive:
                raise SetupError("no installation performed: use --dry-run or explicit --mode with --yes")
            if input("Apply this plan? [y/N]: ").strip().lower() not in {"y", "yes"}:
                print("Cancelled; no installation performed.")
                return 0
        root, result = execute_plan(plan, repository, args.output_dir, args.trace_output,
                                    skip_gpu=args.skip_gpu_check)
        print(json.dumps(result, indent=2))
        print("Prepared files:", root)
        print("Activate the reported target environment, then: source", shlex.quote(str(root / "activate-trace.sh")))
        print("Validate model inference separately before collection; no service was started.")
        return 0
    except (SetupError, OSError, ValueError, KeyError) as exc:
        print(f"Environment preparation failed: {exc}\nNo existing service was stopped. A partially created NEW environment is retained for inspection.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
