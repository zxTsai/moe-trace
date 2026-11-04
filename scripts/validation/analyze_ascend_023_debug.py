"""Verify bounded per-worker JSONL and summarize actual logical expert choices."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def analyze(root, expected_workers=16, allow_incomplete=False):
    statuses = sorted(Path(root).glob("*/dp-*-tp-*/status.json"))
    if not statuses:
        raise ValueError("no worker status.json found; pass output_root/run_id")
    counts, decisions = Counter(), Counter()
    proposals, resolved = set(), set()
    workers, executions, unresolved, limits, pending_verification = [], 0, 0, 0, 0
    identities = set()
    run_ids = set()
    for path in statuses:
        status = json.loads(path.read_text())
        if status["profile"] != "deepseek-v4-023post1-async-graph-debug":
            raise ValueError("wrong worker profile")
        run_ids.add(status["run_id"])
        if len(run_ids) != 1:
            raise ValueError("mixed run IDs")
        identity = (status["ranks"]["dp"], status["ranks"]["tp"])
        if identity[1] != 0 or not 0 <= identity[0] < 16:
            raise ValueError("worker rank outside DP16/TP1")
        if identity in identities:
            raise ValueError("duplicate DP/TP worker identity")
        identities.add(identity)
        if status["state"] == "failed" or (not allow_incomplete and status["state"] != "closed"):
            raise ValueError(f"worker is {status['state']}: {path}; use --allow-incomplete for debug-only partial analysis")
        # Use the indexed prefix when observing a running writer.
        with (path.parent / "events.jsonl").open("rb") as stream:
            data = stream.read(status["bytes"])
        if len(data) != status["bytes"] or hashlib.sha256(data).hexdigest() != status["sha256"]:
            raise ValueError(f"worker data/index hash mismatch: {path}")
        local_executions = set()
        for line in data.splitlines():
            event = json.loads(line)
            if event["type"] == "execution":
                execution = event["execution_id"]
                if execution in local_executions:
                    raise ValueError("duplicate execution ID")
                local_executions.add(execution)
                executions += 1
                if len(event["routes"]) != event["route_records"]:
                    raise ValueError("route count mismatch")
                for row in event["routes"]:
                    key = (event["scope"], row["kind"], row["layer_name"], row["num_experts"])
                    for expert in range(row["num_experts"]):
                        counts[(*key, expert)] += 0
                    for expert in row["expert_ids"]:
                        counts[(*key, expert)] += 1
            elif event["type"] == "sampler":
                if event["execution_id"] not in local_executions:
                    raise ValueError("sampler refers to missing target execution")
                for decision in event["decisions"]:
                    decisions[decision["status"]] += 1
                    unresolved += decision["producer_link"] != "matched"
                    if decision["producer_link"] == "matched":
                        resolved.add((identity, decision["candidate_round"], decision["request_id"], decision["candidate_index"]))
            elif event["type"] == "proposal":
                for candidate in event["candidates"]:
                    proposals.add((identity, event["candidate_round"], candidate["request_id"], candidate["candidate_index"]))
        limits += status["state"] == "limit_reached"
        pending_verification += bool(status.get("pending_verification"))
        workers.append({"path": str(path.parent), "ranks": status["ranks"], "state": status["state"]})
    if not allow_incomplete and len(workers) != expected_workers:
        raise ValueError(f"expected {expected_workers} workers, found {len(workers)}")
    target = sum(n for key,n in counts.items() if key[0] == "target")
    mtp = sum(n for key,n in counts.items() if key[0] == "mtp")
    decisions["unverified"] = len(proposals-resolved)
    for state in ("accepted", "rejected", "not_adopted"):
        decisions[state] += 0
    return {"profile": "deepseek-v4-023post1-async-graph-debug", "npu_verified": False,
            "workers": workers, "executions": executions, "limits_reached": limits,
            "worker_coverage_complete": len(workers) == expected_workers,
            "unresolved_candidate_links": unresolved, "decisions": dict(decisions),
            "capture_observed": executions > 0 and target+mtp > 0,
            "pending_verification_executions": pending_verification,
            "candidate_attribution_complete": unresolved == 0 and pending_verification == 0,
            "expert_choices": {"target": target, "mtp": mtp, "combined": target+mtp},
            "counts": [dict(scope=k[0], kind=k[1], layer_name=k[2], num_experts=k[3], expert_id=k[4], count=n)
                       for k,n in sorted(counts.items())],
            "note": "Counts include rejected computation; logical choices do not measure physical NPU utilization or network traffic."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--expected-workers", type=int, default=16)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = analyze(args.run, args.expected_workers, args.allow_incomplete)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps({k:report[k] for k in ("executions", "expert_choices", "unresolved_candidate_links", "worker_coverage_complete")}))


if __name__ == "__main__":
    main()
