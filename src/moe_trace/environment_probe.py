"""Compatibility script for setup processes started before the directory move."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("setup") / "probe.py"), run_name="__main__")
