"""Catch broken repository navigation and no-dependency entrypoints after moves."""
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest

from moe_trace.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_local_documentation_links_exist():
    documents = [ROOT / "README.md", * (ROOT / "docs").rglob("*.md"),
                 * (ROOT / "patches").rglob("*.md"), ROOT / "configs/README.md",
                 ROOT / "scripts/README.md"]
    broken = []
    for doc in documents:
        for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", doc.read_text(encoding="utf-8")):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            path = unquote(target.split("#", 1)[0])
            if path and not (doc.parent / path).exists():
                broken.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not broken, "\n".join(broken)


@pytest.mark.parametrize("name", ["default", "vllm-0.10.1", "vllm-0.29.0", "ascend-vllm-0.23.0"])
def test_examples_are_portable(name):
    config = load_config(ROOT / "configs/examples" / f"{name}.yaml")
    assert not config.output_root.is_absolute()
    assert config.output_root == Path("outputs")


@pytest.mark.parametrize("entry", ["scripts/prepare_environment.py", "scripts/tools/stage_vllm_029.py",
                                  "scripts/stage_vllm_029.py", "scripts/dev/build_vllm_029_patch.py"])
def test_stdlib_entry_help(entry):
    result = subprocess.run([sys.executable, "-S", str(ROOT / entry), "--help"],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_legacy_probe_remains_runnable_without_dependencies():
    result = subprocess.run([sys.executable, "-S", str(ROOT / "src/moe_trace/environment_probe.py")],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "MOE_TRACE_ENV_JSON=" in result.stdout


@pytest.mark.parametrize("entry", ["reference_requests", "lifecycle_requests", "verify_reference",
                                  "verify_lifecycle", "verify_exports"])
def test_validation_script_help(entry):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/validation" / f"{entry}.py"), "--help"],
                            text=True, capture_output=True, env=env, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
