"""Stage both pinned Ascend packages without importing device dependencies."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[2]
                      / "src/moe_trace/setup/ascend.py"), run_name="__main__")
