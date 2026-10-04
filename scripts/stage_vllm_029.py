"""Compatibility entrypoint; prefer scripts/tools/stage_vllm_029.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("tools") / "stage_vllm_029.py"),
                   run_name="__main__")
