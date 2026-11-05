"""Validate raw ordered token routes, or export one source/window as JSON."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(root,expected_workers=16,allow_incomplete=False,source_dp=None,window_id=None):
    paths=sorted(Path(root).glob("*/source-dp-*/status.json"))
    if source_dp is not None:
        paths=[p for p in paths if json.loads(p.read_text())["ranks"]["dp"]==source_dp]
    if not paths:
        raise ValueError("no routing_trace source folders found")
    sources=[]
    selected=None
    totals={"target":0,"mtp":0}
    all_ep=set()
    runs=set()
    for path in paths:
        status=json.loads(path.read_text())
        if status["profile"]!="deepseek-v4-023post1-routing-trace":
            raise ValueError("wrong routing trace profile")
        if status["state"]=="failed" or (status["state"]!="closed" and not allow_incomplete):
            raise ValueError(f"source is {status['state']}; use --allow-incomplete for partial inspection")
        source=json.loads((path.parent/"worker.json").read_text())
        if any(source[k+"_rank"]!=status["ranks"][k] for k in ("dp","tp","ep")):
            raise ValueError("source identity differs from runtime ranks")
        if source["ep_rank"] in all_ep or not 0<=source["ep_rank"]<16 or source["tp_rank"]!=0:
            raise ValueError("duplicate/unsupported source rank")
        all_ep.add(source["ep_rank"])
        runs.add(status["run_id"])
        if len(runs)!=1:
            raise ValueError("mixed runs")
        placement=json.loads((path.parent/"placements.json").read_text())
        digest=hashlib.sha256(json.dumps(placement,sort_keys=True).encode()).hexdigest()
        seen_windows=set()
        executions=set()
        auxiliary_refs=set()
        missing_devices=set()
        schedules=set()
        tokens=0
        layer_records=0
        for file in path.parent.glob("routes*.jsonl"):
            with file.open() as stream:
                for line in stream:
                    record=json.loads(line)
                    if (record["type"]!="routing_trace_window" or record["source"]!=source
                            or record["run_id"]!=status["run_id"] or record["placement_sha256"]!=digest):
                        raise ValueError("trace identity/placement mismatch")
                    if record["window_id"] in seen_windows:
                        raise ValueError("duplicate trace window")
                    seen_windows.add(record["window_id"])
                    if record["window_id"]==window_id:
                        selected=record
                    for round_ in record["rounds"]:
                        seq=round_["schedule_seq"]
                        if type(seq) is not int or seq<0:
                            raise ValueError("invalid observed schedule sequence")
                        schedules.add(seq)
                        for execution in round_["executions"]:
                            eid=execution["execution_id"]
                            if eid in executions:
                                raise ValueError("duplicate execution ID")
                            executions.add(eid)
                            scope=execution["scope"]
                            if scope not in totals:
                                raise ValueError("unknown model scope")
                            if scope=="mtp" and (type(execution["mtp_step"]) is not int or execution["mtp_step"]<0):
                                raise ValueError("missing MTP execution step")
                            expected={name for name,value in placement.items() if value["scope"]==scope}
                            requests=set()
                            rows=set()
                            for request in execution["requests"]:
                                if request["request_id"] in requests:
                                    raise ValueError("duplicate request within execution")
                                requests.add(request["request_id"])
                                for token in request["tokens"]:
                                    i=token["input_row"]
                                    if type(i) is not int or i<0 or i in rows:
                                        raise ValueError("duplicate/invalid actual input row")
                                    rows.add(i)
                                    tokens+=1
                                    if type(token["token_position"]) is not int or token["token_position"]<0:
                                        raise ValueError("invalid actual token position")
                                    if scope=="mtp" and token["input_sequence_position"]!=token["token_position"]+1:
                                        raise ValueError("MTP shifted sequence position mismatch")
                                    if scope=="target" and token["kind"] not in {"target_decode","target_verify"}:
                                        raise ValueError("target phase mismatch")
                                    if scope=="mtp" and token["kind"]!="mtp_draft":
                                        raise ValueError("MTP phase mismatch")
                                    names=[layer["layer_name"] for layer in token["layers"]]
                                    if len(names)!=len(expected) or set(names)!=expected:
                                        raise ValueError("missing/duplicate model layer for token")
                                    for layer in token["layers"]:
                                        mapping=placement[layer["layer_name"]]
                                        ids=layer["expert_ids"]
                                        if (not ids or len(ids)!=len(set(ids)) or layer["num_experts"]!=mapping["num_experts"]
                                                or any(type(e) is not int or not 0<=e<mapping["num_experts"] for e in ids)):
                                            raise ValueError("invalid ordered Top-K expert IDs")
                                        owners=[mapping["owners"][e] for e in ids]
                                        if layer["destination_ep_ranks"]!=owners:
                                            raise ValueError("ordered Top-K destination mapping mismatch")
                                        for rank in owners:
                                            device=record["device_table"].get(str(rank))
                                            if device is None or device["ep_rank"]!=rank:
                                                raise ValueError("missing destination in device table")
                                            if device["identity_status"]!="resolved":
                                                missing_devices.add(rank)
                                        totals[scope]+=len(ids)
                                        layer_records+=1
                    for event in record["sampler_events"]:
                        auxiliary_refs.add(event["execution_id"])
                    for event in record["proposal_events"]:
                        auxiliary_refs.update(item["producer_execution"] for item in event["candidates"])
        missing_refs=sorted(auxiliary_refs-executions)
        if status["full_history_retained"] and missing_refs:
            raise ValueError("auxiliary event refers to missing execution in complete retained history")
        sources.append(dict(source=source,state=status["state"],retained_windows=len(seen_windows),
            executions=len(executions),observed_schedule_sequences=len(schedules),input_tokens=tokens,
            layer_records=layer_records,unresolved_destination_ranks=sorted(missing_devices),
            unresolved_auxiliary_execution_refs=missing_refs,full_history_retained=status["full_history_retained"]))
    expected=1 if source_dp is not None else expected_workers
    if not allow_incomplete and len(sources)!=expected:
        raise ValueError(f"expected {expected} source workers, found {len(sources)}")
    if window_id is not None:
        if selected is None:
            raise ValueError("requested window is not retained")
        return selected
    return dict(profile="deepseek-v4-023post1-routing-trace",npu_verified=False,
        validation="software_structure_and_placement_not_hardware_acceptance",sources=sources,
        worker_coverage_complete=len(sources)==expected,aggregation="retained_windows_only",
        expert_choices=dict(totals,combined=sum(totals.values())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-dp",type=int)
    parser.add_argument("--window",type=int)
    parser.add_argument("--expected-workers",type=int,default=16)
    parser.add_argument("--allow-incomplete",action="store_true")
    args=parser.parse_args()
    if args.window is not None and args.source_dp is None:
        parser.error("--window requires --source-dp")
    if args.output.suffix==".jsonl" or args.output.name in {"status.json","worker.json","placements.json"}:
        parser.error("choose a JSON export path that does not overwrite trace data or metadata")
    try:
        report=analyze(args.run,args.expected_workers,args.allow_incomplete,args.source_dp,args.window)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    except (ValueError,OSError,KeyError) as exc:
        parser.exit(2,f"{exc}\n")
    print(args.output)


if __name__=="__main__":
    main()
