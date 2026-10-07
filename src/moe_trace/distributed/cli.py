"""Offline commands for the experimental distributed Decode contract."""
import argparse
import json
import sys

from ..errors import IntegrityError
from .storage import collect_run, export_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("run_dir")
    merge = sub.add_parser("merge")
    merge.add_argument("run_dir")
    merge.add_argument("output")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            manifest, joined = collect_run(args.run_dir)
            print(json.dumps({"contract_complete": True, "npu_verified": False,
                              "validation_level": manifest["validation_level"],
                              "logical_routes": len(joined)}, indent=2))
        else:
            count = export_run(args.run_dir, args.output)
            print(json.dumps({"logical_routes": count, "output": args.output,
                              "npu_verified": False}))
    except (IntegrityError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"contract_complete": False, "error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
