"""Bootstrap without importing moe_trace (and therefore without its dependencies)."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/moe_trace/environment.py"),
              run_name="__main__")
