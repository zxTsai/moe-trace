"""Stage checked vLLM 0.29.0 patches without changing the installed package."""
import argparse
import runpy
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="installed vLLM package directory")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    setup = runpy.run_path(str(repository / "src/moe_trace/setup/manager.py"))
    root = Path(tempfile.mkdtemp(prefix="moe-trace-vllm-029-"))
    print(setup["stage_package"](args.package.resolve(), "0.29.0", repository, root))


if __name__ == "__main__":
    main()
