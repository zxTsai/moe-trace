"""Summarize retained load windows, with explicit rolling-history coverage."""
import argparse
from collections import Counter
import json
from pathlib import Path


def analyze(root,expected_workers=16,allow_incomplete=False):
    paths=sorted(Path(root).glob("*/dp-*-tp-*/status.json"))
    if not paths:
        raise ValueError("no summary workers found; pass output_root/run_id")
    counts=Counter()
    workers=[]
    ranks=set()
    windows=0
    total={"target":0,"mtp":0}
    for path in paths:
        status=json.loads(path.read_text())
        if status["profile"]!="deepseek-v4-023post1-load-summary":
            raise ValueError("not a periodic load summary run")
        if status["state"]=="failed" or (status["state"]!="closed" and not allow_incomplete):
            raise ValueError(f"worker is {status['state']}; stop workers or use --allow-incomplete for live inspection")
        rank=(status["ranks"]["dp"],status["ranks"]["tp"])
        if rank in ranks or rank[1]!=0 or not 0<=rank[0]<16:
            raise ValueError("duplicate or unsupported worker rank")
        ranks.add(rank)
        seen=set()
        for file in path.parent.glob("load*.jsonl"):
            with file.open() as stream:
                for line in stream:
                    record=json.loads(line)
                    if (record["type"]!="load_summary" or record["sampling"]!="continuous"
                            or record["run_id"]!=status["run_id"] or record["ranks"]!=status["ranks"]):
                        raise ValueError("window identity mismatch")
                    if record["window_id"] in seen:
                        raise ValueError("duplicate window ID")
                    seen.add(record["window_id"])
                    calculated={"target":0,"mtp":0}
                    for row in record["layers"]:
                        values=row["expert_selection_counts"]
                        if (len(values)!=row["num_experts"] or any(type(n) is not int or n<0 for n in values)
                                or sum(values)!=row["observed_input_tokens"]*row["top_k"]):
                            raise ValueError("invalid expert count vector or token/Top-K totals")
                        calculated[row["scope"]]+=sum(values)
                        for expert,n in enumerate(values):
                            counts[(row["scope"],row["kind"],row["layer_name"],expert)]+=n
                    if record["expert_choices"]!=dict(calculated,combined=sum(calculated.values())):
                        raise ValueError("window totals mismatch")
                    for scope,n in calculated.items():
                        total[scope]+=n
        windows+=len(seen)
        workers.append(dict(ranks=status["ranks"],state=status["state"],rotations=status["rotations"],
            full_history_retained=status["full_history_retained"],retained_windows=len(seen),
            first_window=min(seen) if seen else None,last_window=max(seen) if seen else None))
    if not allow_incomplete and len(workers)!=expected_workers:
        raise ValueError(f"expected {expected_workers} workers, found {len(workers)}")
    return dict(profile="deepseek-v4-023post1-load-summary",npu_verified=False,
        aggregation="retained_windows_only",retained_windows=windows,workers=workers,
        worker_coverage_complete=len(workers)==expected_workers,
        full_history_retained=all(w["full_history_retained"] for w in workers),
        expert_choices=dict(total,combined=sum(total.values())),
        counts=[dict(scope=k[0],kind=k[1],layer_name=k[2],expert_id=k[3],count=n) for k,n in sorted(counts.items())])


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
    print(json.dumps({k:report[k] for k in ("retained_windows","expert_choices","full_history_retained")}))


if __name__=="__main__":
    main()
