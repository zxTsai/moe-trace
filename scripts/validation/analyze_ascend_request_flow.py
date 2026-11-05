"""Validate and aggregate retained per-request DP-to-EP logical routing windows."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def analyze(root,expected_workers=16,allow_incomplete=False):
    statuses=sorted(Path(root).glob("*/dp-*-tp-*/status.json"))
    if not statuses:
        raise ValueError("no request-flow workers found")
    requests={}
    workers=[]
    seen_ranks=set()
    run_ids=set()
    missing=set()
    totals=Counter(target=0,mtp=0)
    for path in statuses:
        status=json.loads(path.read_text())
        if status["profile"]!="deepseek-v4-023post1-request-flow":
            raise ValueError("wrong flow profile")
        if status["state"]=="failed" or (status["state"]!="closed" and not allow_incomplete):
            raise ValueError(f"worker is {status['state']}")
        source=json.loads((path.parent/"worker.json").read_text())
        if any(source[name+"_rank"]!=status["ranks"][name] for name in ("dp","tp","ep")):
            raise ValueError("worker metadata differs from runtime ranks")
        placement=json.loads((path.parent/"placements.json").read_text())
        digest=hashlib.sha256(json.dumps(placement,sort_keys=True).encode()).hexdigest()
        rank=source["ep_rank"]
        if rank in seen_ranks or not 0<=rank<16:
            raise ValueError("duplicate or unsupported EP rank")
        seen_ranks.add(rank)
        run_ids.add(status["run_id"])
        if len(run_ids)!=1:
            raise ValueError("mixed run IDs")
        seen_windows=set()
        for file in path.parent.glob("load*.jsonl"):
            with file.open() as stream:
                for line in stream:
                    record=json.loads(line)
                    if (record["type"]!="request_flow_summary" or record["placement_sha256"]!=digest
                            or record["run_id"]!=status["run_id"] or record["ranks"]!=status["ranks"]):
                        raise ValueError("window identity/placement mismatch")
                    if record["window_id"] in seen_windows:
                        raise ValueError("duplicate flow window")
                    seen_windows.add(record["window_id"])
                    window=Counter(target=0,mtp=0)
                    seen_requests=set()
                    for request in record["requests"]:
                        req=request["request_id"]
                        if request["source"]!=source or req in seen_requests:
                            raise ValueError("request source mismatch or duplicate request")
                        seen_requests.add(req)
                        key=(rank,req)
                        target=requests.setdefault(key,dict(request_id=req,source=source,counts=Counter()))
                        for layer in request["layers"]:
                            mapping=placement[layer["layer_name"]]
                            if mapping["scope"]!=layer["scope"]:
                                raise ValueError("model scope differs from placement")
                            n=0
                            selected=set()
                            for flow in layer["destinations"]:
                                destination=flow["destination"]["ep_rank"]
                                if flow["destination"]["identity_status"]!="resolved":
                                    missing.add(destination)
                                count=0
                                for expert in flow["expert_counts"]:
                                    eid,value=expert["expert_id"],expert["count"]
                                    if (type(eid) is not int or not 0<=eid<mapping["num_experts"]
                                            or mapping["owners"][eid]!=destination or eid in selected
                                            or type(value) is not int or value<1):
                                        raise ValueError("expert owner/count mismatch")
                                    selected.add(eid)
                                    count+=value
                                    target["counts"][(layer["scope"],layer["kind"],layer["layer_name"],destination,eid)]+=value
                                if count!=flow["expert_selection_count"]:
                                    raise ValueError("destination count mismatch")
                                n+=count
                            if n!=layer["observed_input_tokens"]*layer["top_k"]:
                                raise ValueError("request token/Top-K totals mismatch")
                            window[layer["scope"]]+=n
                    if record["expert_choices"]!=dict(window,combined=sum(window.values())):
                        raise ValueError("window totals mismatch")
                    totals.update(window)
        workers.append(dict(ranks=status["ranks"],state=status["state"],retained_windows=len(seen_windows),
            full_history_retained=status["full_history_retained"]))
    if not allow_incomplete and len(workers)!=expected_workers:
        raise ValueError(f"expected {expected_workers} workers, found {len(workers)}")
    result=[]
    for key,value in sorted(requests.items()):
        result.append(dict(request_id=value["request_id"],source=value["source"],
            flows=[dict(scope=k[0],kind=k[1],layer_name=k[2],destination_ep_rank=k[3],expert_id=k[4],
                        expert_selection_count=n) for k,n in sorted(value["counts"].items())]))
    return dict(profile="deepseek-v4-023post1-request-flow",npu_verified=False,
        metric="expert_selection_count_not_network_bytes",aggregation="retained_windows_only",
        requests=result,workers=workers,worker_coverage_complete=len(workers)==expected_workers,
        full_history_retained=all(w["full_history_retained"] for w in workers),
        unresolved_destination_ranks=sorted(missing),expert_choices=dict(totals,combined=sum(totals.values())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-workers",type=int,default=16)
    parser.add_argument("--allow-incomplete",action="store_true")
    args=parser.parse_args()
    try:
        report=analyze(args.run,args.expected_workers,args.allow_incomplete)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    except (ValueError,OSError,KeyError) as exc:
        parser.exit(2,f"{exc}\n")
    print(json.dumps(dict(requests=len(report["requests"]),expert_choices=report["expert_choices"],
        unresolved_destination_ranks=report["unresolved_destination_ranks"])))


if __name__=="__main__":
    main()
