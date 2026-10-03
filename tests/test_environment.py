import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from moe_trace import environment as env


@pytest.fixture
def fixture(tmp_path):
    repo = tmp_path / "repo"
    package = tmp_path / "installed"
    package.mkdir()
    (package / "module.py").write_text("original\n")
    (package / "native.so").write_bytes(b"native")
    manifest = repo / "patches/vllm/0.29.0"
    manifest.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("")
    (manifest / "base_files.json").write_text(json.dumps({"sha256": {
        "module.py": hashlib.sha256(b"original\n").hexdigest()}}))
    (manifest / "0001-trace-hooks.patch").write_text(
        "diff --git a/module.py b/module.py\n--- a/module.py\n+++ b/module.py\n@@ -1 +1 @@\n-original\n+patched\n")
    report = dict(python="/target/bin/python", prefix="/target", python_version="3.12.0",
                  platform="linux", machine="x86_64", vllm_package=str(package),
                  packages={"vllm": "0.29.0", "torch": "2.13.0", "transformers": "5.18.0",
                            "numpy": "2.3.5", "PyYAML": "6.0.3", "setuptools": "80.0"},
                  gpu={"verified": True})
    return repo, package, report


def test_compatible(fixture):
    repo, _, report = fixture
    assert env.assess(report, repo)["compatible"]


@pytest.mark.parametrize("key,value", [("vllm", None), ("vllm", "0.30.0"),
    ("torch", "2.12.0"), ("transformers", None), ("numpy", "1.25"),
    ("PyYAML", "5.0"), ("setuptools", "67.0")])
def test_incompatible_dependencies(fixture, key, value):
    repo, _, report = fixture
    report["packages"][key] = value
    assert not env.assess(report, repo)["compatible"]


@pytest.mark.parametrize("key,value", [("platform", "win32"), ("machine", "arm64"),
                                      ("python_version", "3.10.0")])
def test_platform_rejected(fixture, key, value):
    repo, _, report = fixture
    report[key] = value
    assert not env.assess(report, repo)["compatible"]


def test_source_mismatch(fixture):
    repo, package, report = fixture
    (package / "module.py").write_text("user change")
    assert not env.assess(report, repo)["compatible"]
    with pytest.raises(env.SetupError, match="base differs"):
        env.stage_package(package, "0.29.0", repo, repo / "output")
    assert (package / "module.py").read_text() == "user change"


@pytest.mark.parametrize("name", ["base", "root", "existing", "../escape", "", "bad name"])
def test_reserved_or_existing_targets(name):
    with pytest.raises(env.SetupError):
        env.validate_new_target(name, None, [Path("/envs/existing")])


def test_existing_prefix_rejected(tmp_path):
    with pytest.raises(env.SetupError):
        env.validate_new_target(None, tmp_path, [])


@pytest.mark.parametrize("version", list(env.PROFILES))
def test_new_plan(version, tmp_path):
    plan = env.build_plan("new", version, tmp_path, [], conda="conda", target=["--name", "fresh"])
    assert f"python={env.PROFILES[version]['python']}" in plan["commands"][0]
    assert f"vllm=={version}" in plan["commands"][1]
    assert "--only-binary=:all:" in plan["commands"][1]
    assert all("serve" not in command for command in plan["commands"])


def test_existing_plan_never_installs_dependencies(tmp_path):
    plan = env.build_plan("existing", "0.29.0", tmp_path, ["python"])
    assert len(plan["commands"]) == 1
    assert "--no-deps" in plan["commands"][0]
    assert "--no-build-isolation" in plan["commands"][0]


def test_real_patch_overlay_preserves_original(fixture, tmp_path):
    repo, package, _ = fixture
    overlay = env.stage_package(package, "0.29.0", repo, tmp_path / "prepared")
    assert (overlay / "vllm/module.py").read_text() == "patched\n"
    assert (package / "module.py").read_text() == "original\n"
    assert (overlay / "vllm/native.so").is_symlink()


def test_artifacts_and_wrong_environment_guard(fixture, tmp_path):
    repo, _, report = fixture
    result = env.write_artifacts(tmp_path, repo, report, {"version": "0.29.0"},
                                 tmp_path / "overlay", tmp_path / "space output", False)
    assert result["status"] == "prepared_gpu_unverified"
    assert not result["service_started"]
    script = tmp_path / "activate-trace.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert subprocess.run(["bash", "-c", f"source '{script}'"], capture_output=True).returncode == 1
    import yaml
    assert yaml.safe_load((tmp_path / "trace.yaml").read_text())["trace"]["enabled"]


@pytest.mark.parametrize("arguments", [["--detect"], ["--mode", "new", "--dry-run"],
                                      ["--yes"], ["--mode", "new"]])
def test_inspection_and_unapproved_calls_do_not_install(fixture, monkeypatch, arguments):
    repo, _, report = fixture
    monkeypatch.setattr(env, "probe", lambda *a, **k: report)
    monkeypatch.setattr(env, "conda_inventory", lambda *a: [])
    monkeypatch.setattr(env.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(env, "execute_plan", lambda *a, **k: pytest.fail("unexpected installation"))
    result = env.main(["--repository", str(repo), "--conda", "conda", *arguments])
    assert result == (0 if "--detect" in arguments or "--dry-run" in arguments else 1)


def test_existing_gpu_failure_stops_before_overlay(fixture, monkeypatch, tmp_path):
    repo, _, report = fixture
    report["gpu"] = {"verified": False}
    monkeypatch.setattr(env, "probe", lambda *a, **k: report)
    monkeypatch.setattr(env, "run", lambda *a, **k: "")
    plan = env.build_plan("existing", "0.29.0", repo, ["python"])
    with pytest.raises(env.SetupError, match="GPU/native"):
        env.execute_plan(plan, repo, tmp_path / "outputs", tmp_path / "traces")
    assert not (tmp_path / "outputs").exists()


def test_existing_preparation_workflow(fixture, monkeypatch, tmp_path):
    repo, package, report = fixture
    monkeypatch.setattr(env, "probe", lambda *a, **k: report)
    original_run = env.run
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return "" if "pip" in command or "-c" in command else original_run(command, **kwargs)
    monkeypatch.setattr(env, "run", run)
    plan = env.build_plan("existing", "0.29.0", repo, ["python"])
    root, result = env.execute_plan(plan, repo, tmp_path / "outputs", tmp_path / "traces")
    assert result["gpu_verified"]
    assert (root / "environment.json").exists()
    assert (package / "module.py").read_text() == "original\n"
    assert sum("install" in command for command in commands) == 1


def test_install_failure_stops(fixture, monkeypatch, tmp_path):
    repo, _, _ = fixture
    monkeypatch.setattr(env, "conda_inventory", lambda *a: [])
    seen = []
    def fail(command, **kwargs):
        seen.append(command)
        raise env.SetupError("installer failed")
    monkeypatch.setattr(env, "run", fail)
    plan = env.build_plan("new", "0.29.0", repo, [], conda="conda", target=["--name", "fresh"])
    with pytest.raises(env.SetupError, match="installer failed"):
        env.execute_plan(plan, repo, tmp_path / "outputs", tmp_path / "traces")
    assert len(seen) == 1
    assert not (tmp_path / "outputs").exists()


def test_new_preparation_workflow(fixture, monkeypatch, tmp_path):
    repo, _, report = fixture
    monkeypatch.setattr(env, "conda_inventory", lambda *a: [])
    monkeypatch.setattr(env, "probe", lambda *a, **k: report)
    original_run = env.run
    seen = []
    def run(command, **kwargs):
        seen.append(command)
        return "" if command[0] == "fake-conda" else original_run(command, **kwargs)
    monkeypatch.setattr(env, "run", run)
    plan = env.build_plan("new", "0.29.0", repo, [], conda="fake-conda", target=["--name", "fresh"])
    _, result = env.execute_plan(plan, repo, tmp_path / "outputs", tmp_path / "traces", skip_gpu=True)
    assert seen[:3] == plan["commands"]
    assert result["status"] == "prepared_gpu_unverified"


def test_interactive_cancellation(fixture, monkeypatch):
    repo, _, report = fixture
    monkeypatch.setattr(env, "probe", lambda *a, **k: report)
    monkeypatch.setattr(env.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    monkeypatch.setattr(env, "execute_plan", lambda *a, **k: pytest.fail("installation after cancellation"))
    assert env.main(["--repository", str(repo), "--mode", "existing"]) == 0
