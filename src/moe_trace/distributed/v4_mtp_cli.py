"""Validate V4 MTP producers, verification outcomes and every worker's routes."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from ..errors import IntegrityError
from ..io import atomic_json
from .storage import collect_run

PROFILE = "deepseek-v4-mtp-debug"


def ledger(directory, name):
    data = (directory / f"{name}.jsonl").read_bytes()
    index = json.loads((directory / f"{name}-index.json").read_text())
    values = [json.loads(line) for line in data.splitlines()]
    if hashlib.sha256(data).hexdigest() != index["sha256"] or len(values) != index["records"]:
        raise IntegrityError(f"invalid {name} ledger hash/count")
    return values


def collect(root):
    root = Path(root)
    expected = {f"dp-{d}-tp-0" for d in range(16)}
    deployment = json.loads((root / "deployment.json").read_text())
    if (deployment["profile"] != PROFILE or deployment["run_id"] != root.name
            or len(deployment["expected_workers"]) != 16 or set(deployment["expected_workers"]) != expected
            or {p.name for p in root.iterdir() if p.is_dir()} != expected):
        raise IntegrityError("invalid MTP deployment topology")
    all_routes, all_candidates, layers = [], [], None
    for dp in range(16):
        name = f"dp-{dp}-tp-0"
        directory = root / name
        inputs, produced = ledger(directory, "inputs"), ledger(directory, "candidates")
        manifest, joined = collect_run(directory / "trace", allow_empty=True)
        if manifest["validation_level"] != "a3_unverified_debug" or len(manifest["workers"]) != 1:
            raise IntegrityError("invalid MTP worker manifest")
        worker = manifest["workers"][0]
        if (worker["deployment"] != PROFILE or worker["run"] != root.name
                or worker["node"] != deployment["node_id"] or worker["worker"] != name
                or (worker["dp_rank"], worker["tp_rank"], worker["ep_rank"], worker["device"]) != (dp, 0, dp, dp)):
            raise IntegrityError("MTP worker identity mismatch")
        if layers is not None and layers != manifest["model_layers"]:
            raise IntegrityError("workers have different model scopes/layers")
        layers = manifest["model_layers"]
        if set(layers) != {"target", "mtp"} or layers["mtp"] != [0]:
            raise IntegrityError("unsupported MTP layer declarations")
        declarations, expected_keys = {}, set()
        for declaration in inputs:
            scope, execution = declaration["scope"], declaration["execution"]
            if scope not in layers or (scope, execution) in declarations:
                raise IntegrityError("duplicate or invalid execution declaration")
            rows = {r["input_row"]: r for r in declaration["rows"]}
            if len(rows) != len(declaration["rows"]):
                raise IntegrityError("duplicate MTP input row")
            declarations[scope, execution] = rows
            expected_keys.update((scope, execution, i, layer) for i in rows for layer in layers[scope])
        engine = f"D-v4-mtp-dp-{dp}"
        seen, candidate_map, verifications = set(), {}, {}
        for candidate in produced:
            candidate.setdefault("round_execution", candidate["producer_execution"])
            candidate.setdefault("proposal_k", 1)  # original K=1 development traces
            key = (candidate["request"], candidate["candidate_round"], candidate["candidate_node"])
            execution, i = candidate["producer_execution"], candidate["producer_input_row"]
            row = declarations.get(("mtp", execution), {}).get(i)
            if (key in candidate_map or row is None or row["kind"] != "mtp_draft"
                    or row["request"] != key[0] or not candidate["candidate_node"].isdigit()
                    or candidate["candidate_round"] != candidate["round_execution"].replace("execution-", "draft-", 1)
                    or candidate["position"] != row["position"] + 2
                    or candidate["producer_kernel_position"] != row["position"]):
                raise IntegrityError("invalid MTP producer-to-candidate association")
            candidate_map[key] = candidate
        for route, accepted in joined:
            row, scope = route.row, route.model_scope
            key = (scope, row.execution, row.input_row, route.layer)
            declared = declarations.get((scope, row.execution), {}).get(row.input_row)
            actual = dict(input_row=row.input_row, request=row.local_request, position=row.position,
                          kind=row.kind, candidate_round=row.candidate_round, candidate_node=row.candidate_node)
            if (key in seen or declared != actual or row.source_engine != engine
                    or row.global_request != f"{engine}:{row.local_request}"
                    or (scope == "mtp" and row.kind != "mtp_draft")
                    or (scope == "target" and row.kind not in {"target_decode", "target_verify"})):
                raise IntegrityError("route differs from actual scoped MTP input")
            seen.add(key)
            if row.kind == "target_verify" and row.candidate_node is not None:
                ckey = (row.local_request, row.candidate_round, row.candidate_node)
                candidate = candidate_map.get(ckey)
                if (candidate is None or candidate["position"] != row.position or accepted is None
                        or int(candidate["producer_execution"].split("-")[-1]) >= int(row.execution.split("-")[-1])
                        or (candidate["token_id"] is not None and row.token_id != candidate["token_id"])):
                    raise IntegrityError("verification differs from candidate producer")
                decision = (row.execution, row.input_row, accepted)
                if ckey in verifications and verifications[ckey] != decision:
                    raise IntegrityError("candidate verified more than once or inconsistent outcome")
                verifications[ckey] = decision
            all_routes.append((route, accepted))
        if seen != expected_keys:
            raise IntegrityError("missing MTP/target input-layer coverage")
        groups = {}
        for key in candidate_map:
            groups.setdefault(key[:2], []).append(int(key[2]))
        for group, nodes in groups.items():
            if sorted(nodes) != list(range(len(nodes))):
                raise IntegrityError("missing or duplicate MTP producer node")
            produced_group = [candidate_map[*group, str(i)] for i in range(len(nodes))]
            if (any(c["proposal_k"] != len(nodes) for c in produced_group)
                    or produced_group[0]["round_execution"] != produced_group[0]["producer_execution"]):
                raise IntegrityError("MTP output ledger lacks configured producer nodes")
            if any(c["position"] != produced_group[0]["position"]+i for i,c in enumerate(produced_group)):
                raise IntegrityError("MTP candidate positions are not autoregressive")
            verified = [i for i in range(len(nodes)) if (*group,str(i)) in verifications]
            if verified != list(range(len(verified))):
                raise IntegrityError("verification is not a candidate prefix")
            decisions = [verifications[*group,str(i)] for i in verified]
            if any(a[0] != b[0] or a[1]+1 != b[1] for a,b in zip(decisions,decisions[1:])):
                raise IntegrityError("candidate verification inputs are not contiguous")
            flags = [d[2] for d in decisions]
            if flags != sorted(flags,reverse=True):
                raise IntegrityError("acceptance results are not a prefix")
        for key, candidate in candidate_map.items():
            decision = verifications.get(key)
            earlier_rejection = any(verifications.get((*key[:2],str(i)), (None,None,True))[2] is False
                                    for i in range(int(key[2])))
            all_candidates.append({**candidate, "source_engine": engine,
                "verification_execution": decision[0] if decision else None,
                "accepted": decision[2] if decision else None,
                "state": ("accepted" if decision[2] else "not_adopted" if earlier_rejection else "rejected")
                         if decision else "unverified"})
    if {r.model_scope for r, _ in all_routes} != {"target", "mtp"}:
        raise IntegrityError("need actual target and MTP routes")
    return all_routes, all_candidates


def route_decisions(joined):
    flags = {}
    for route, accepted in joined:
        row = route.row
        if row.candidate_node is not None:
            flags[row.global_request,row.candidate_round,row.candidate_node] = accepted
    decisions = {}
    for key, flag in flags.items():
        earlier = any(flags.get((*key[:2],str(i))) is False for i in range(int(key[2])))
        decisions[key] = "accepted" if flag is True else "not_adopted" if earlier else "rejected"
    return decisions


def analyze(joined):
    totals = dict(target=0, mtp=0, combined=0)
    counts, sizes, executions = Counter(), {}, Counter()
    decisions = route_decisions(joined)
    for route, accepted in joined:
        scope = route.model_scope
        totals[scope] += len(route.expert_ids)
        totals["combined"] += len(route.expert_ids)
        if (scope,route.layer) in sizes and sizes[scope,route.layer] != route.num_experts:
            raise IntegrityError("inconsistent expert count across workers")
        sizes[scope, route.layer] = route.num_experts
        decision = decisions.get((route.row.global_request,route.row.candidate_round,route.row.candidate_node),"not_applicable")
        for expert in route.expert_ids:
            counts[scope, route.row.kind, route.layer, decision, expert] += 1
            executions[route.row.source_engine, route.row.execution, scope, route.row.kind, route.layer, expert] += 1
    return totals, counts, sizes, executions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--counts", type=Path)
    parser.add_argument("--by-execution", type=Path)
    args = parser.parse_args()
    try:
        paths = [p.resolve() for p in (args.output, args.counts, args.by_execution) if p]
        if len(set(paths)) != len(paths) or any(p.exists() or p.is_relative_to(args.run_dir.resolve()) for p in paths):
            raise IntegrityError("exports must be distinct new paths outside raw run")
        joined, candidates = collect(args.run_dir)
        totals, counts, sizes, executions = analyze(joined)
        decisions = route_decisions(joined)
        if args.output:
            atomic_json(args.output, {"profile": PROFILE, "npu_verified": False,
                "validation_level": "a3_unverified_debug", "requests_terminal_verified": False,
                "decision_semantics": "rejection sampler decision; not client-visible EOS/stop clipping",
                "expert_choices": totals, "candidates": candidates,
                "routes": [{**asdict(r), "candidate_accepted": a,
                    "candidate_decision": decisions.get((r.row.global_request,r.row.candidate_round,r.row.candidate_node))}
                           for r, a in joined]})
        if args.counts:
            groups = sorted({key[:4] for key in counts})
            with args.counts.open("x", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["model_scope", "execution_kind", "layer", "candidate_decision", "expert_id", "choices"])
                writer.writerows((*group, e, counts[*group, e])
                    for group in groups for e in range(sizes[group[0], group[2]]))
        if args.by_execution:
            with args.by_execution.open("x", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["source_engine", "execution", "model_scope", "execution_kind", "layer", "expert_id", "choices"])
                writer.writerows((*key, n) for key, n in sorted(executions.items()))
        print(json.dumps({"worker_traces_complete": True, "logical_routes": len(joined),
                          "expert_choices": totals, "candidate_states": dict(Counter(c["state"] for c in candidates)),
                          "npu_verified": False}))
    except (IntegrityError, ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps({"worker_traces_complete": False, "error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
