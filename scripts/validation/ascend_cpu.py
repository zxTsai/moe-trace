"""Run moe-trace CPU simulation, not NPU inference or the upstream Ascend UT suite."""
from moe_trace.validation.ascend_cpu import main

if __name__ == "__main__":
    raise SystemExit(main())
