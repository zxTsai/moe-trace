from pathlib import Path

import pytest

from moe_trace.config import ConfigError, load_config


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_are_disabled(tmp_path):
    config = load_config(write(tmp_path / "config.yaml", "{}\n"))
    assert not config.enabled
    assert config.buffer_records == 8192


@pytest.mark.parametrize("body", [
    "trace: {mystery: 1}\n",
    "other: {}\n",
    "trace: {phases: [wrong]}\n",
    "trace: {buffer_records: 2, shard_records: 3}\n",
])
def test_rejects_unknown_or_invalid_values(tmp_path, body):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path / "config.yaml", body))

