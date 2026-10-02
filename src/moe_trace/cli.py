from __future__ import annotations

import argparse
import json

from .analysis import write_counts_csv, write_details_csv, write_details_json
from .reader import TraceReader


def main() -> int:
    parser = argparse.ArgumentParser(prog="moe-trace")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("run_dir")
    for name in ("counts", "details"):
        cmd = sub.add_parser(name)
        cmd.add_argument("run_dir")
        cmd.add_argument("output")
        cmd.add_argument("--request-id")
        cmd.add_argument("--step-id", type=int)
        if name == "details":
            cmd.add_argument("--format", choices=("csv", "json"), default="csv")
    args = parser.parse_args()
    reader = TraceReader(args.run_dir)
    if args.command == "validate":
        issues = reader.validate()
        print(json.dumps({"complete": not issues, "issues": issues}, ensure_ascii=False, indent=2))
        return 1 if issues else 0
    filters = {"request_id": args.request_id, "step_id": args.step_id}
    if args.command == "counts":
        write_counts_csv(reader, args.output, **filters)
    elif args.format == "json":
        write_details_json(reader, args.output, **filters)
    else:
        write_details_csv(reader, args.output, **filters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
