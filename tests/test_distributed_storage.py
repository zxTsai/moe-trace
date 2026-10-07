from dataclasses import asdict, replace
import json

import pytest

from moe_trace.distributed.decode import Worker, DecodeRow, Route, CandidateResult
from moe_trace.distributed.storage import create_run, WorkerWriter, collect_run, export_run
from moe_trace.errors import IntegrityError


def fixtures(tmp_path, layers=None):
    owner = Worker("deploy", "run", "D0", "w0", 0, 0, 0, 0)
    peer = replace(owner, worker="w1", tp_rank=1, ep_rank=1, device=1)
    row = DecodeRow("engine", "execution", 4, "request", "local", 0, 7, 9,
                    "target_verify", "round", "candidate")
    route = Route(owner, owner, row, "target", 0, 4, (3, 1), (0.75, 0.25))
    result = CandidateResult("request", "engine", "round", "candidate", False)
    root = create_run(tmp_path / "run", [owner, peer], layers or {"target": [0]})
    return root, route, replace(route, observer=peer), result


def write_fixture(root, route, peer, result, *, shard_records=2):
    writer = WorkerWriter(root, route.owner, shard_records)
    writer.declare_execution("target", [route.row], [route.owner, peer.observer])
    writer.append(route)
    writer.append(result)
    writer.close()
    writer = WorkerWriter(root, peer.observer, shard_records)
    writer.append(peer)
    writer.close()


def test_shards_roundtrip_and_atomic_overwrite_analysis(tmp_path):
    root, route, peer, result = fixtures(tmp_path)
    write_fixture(root, route, peer, result)
    manifest, joined = collect_run(root)
    assert manifest["validation_level"] == "cpu_contract_only"
    assert joined == [(route, False)]
    assert len(list((root / "worker-00000").glob("events-*.jsonl"))) == 2
    output = tmp_path / "analysis.json"
    for _ in range(2):
        assert export_run(root, output) == 1
    payload = json.loads(output.read_text())
    assert payload["routes"][0]["expert_ids"] == [3, 1]
    assert payload["routes"][0]["candidate_accepted"] is False
    assert not list(tmp_path.glob("*.pending"))


def test_existing_run_and_worker_are_not_overwritten(tmp_path):
    root, route, _, _ = fixtures(tmp_path)
    with pytest.raises(FileExistsError):
        create_run(root, [route.owner], {"target": [0]})
    WorkerWriter(root, route.owner)
    with pytest.raises(FileExistsError):
        WorkerWriter(root, route.owner)


def test_missing_worker_and_running_worker_fail(tmp_path):
    root, route, peer, result = fixtures(tmp_path)
    writer = WorkerWriter(root, route.owner)
    writer.declare_execution("target", [route.row])
    writer.append(route)
    writer.append(result)
    writer.close()
    with pytest.raises(IntegrityError, match="workers"):
        collect_run(root)
    WorkerWriter(root, peer.observer)
    with pytest.raises(IntegrityError, match="incomplete"):
        collect_run(root)


def test_wrong_observer_marks_writer_failed(tmp_path):
    root, route, peer, _ = fixtures(tmp_path)
    writer = WorkerWriter(root, route.owner)
    with pytest.raises(IntegrityError, match="observer"):
        writer.append(peer)
    assert json.loads((writer.directory / "status.json").read_text())["state"] == "failed"
    with pytest.raises(IntegrityError, match="failed"):
        writer.close()


def test_checksum_corruption_fails(tmp_path):
    root, route, peer, result = fixtures(tmp_path)
    write_fixture(root, route, peer, result)
    shard = root / "worker-00000/events-000000.jsonl"
    shard.write_bytes(shard.read_bytes() + b"\n")
    with pytest.raises(IntegrityError, match="checksum"):
        collect_run(root)


def test_missing_layer_fails(tmp_path):
    root, route, peer, result = fixtures(tmp_path, {"target": [0, 1]})
    write_fixture(root, route, peer, result)
    with pytest.raises(IntegrityError, match="layers"):
        collect_run(root)


def test_missing_entire_input_fails(tmp_path):
    root, route, peer, result = fixtures(tmp_path)
    writer = WorkerWriter(root, route.owner)
    writer.declare_execution("target", [route.row, replace(route.row, input_row=1)],
                             [route.owner, peer.observer])
    writer.append(route)
    writer.append(result)
    writer.close()
    writer = WorkerWriter(root, peer.observer)
    writer.append(peer)
    writer.close()
    with pytest.raises(IntegrityError, match="declared inputs"):
        collect_run(root)


def test_missing_candidate_acceptance_fails(tmp_path):
    root, route, peer, _ = fixtures(tmp_path)
    writer = WorkerWriter(root, route.owner)
    writer.declare_execution("target", [route.row])
    writer.append(route)
    writer.close()
    WorkerWriter(root, peer.observer).close()
    with pytest.raises(IntegrityError, match="acceptance"):
        collect_run(root)


def test_no_declaration_is_not_complete(tmp_path):
    root, route, peer, result = fixtures(tmp_path)
    writer = WorkerWriter(root, route.owner)
    writer.append(route)
    writer.append(result)
    writer.close()
    WorkerWriter(root, peer.observer).close()
    with pytest.raises(IntegrityError, match="declared inputs"):
        collect_run(root)


def test_analysis_cannot_be_written_into_run(tmp_path):
    root, route, peer, result = fixtures(tmp_path)
    write_fixture(root, route, peer, result)
    with pytest.raises(IntegrityError, match="outside"):
        export_run(root, root / "analysis.json")


def test_uncommitted_pending_file_fails(tmp_path):
    root, route, peer, result = fixtures(tmp_path)
    write_fixture(root, route, peer, result)
    (root / "worker-00000/.events-000002.jsonl.pending").write_text("partial")
    with pytest.raises(IntegrityError, match="files"):
        collect_run(root)


def test_write_error_cannot_mark_complete(tmp_path, monkeypatch):
    root, route, _, _ = fixtures(tmp_path)
    writer = WorkerWriter(root, route.owner, 1)
    def fail():
        raise OSError("disk full")
    monkeypatch.setattr(writer, "_flush", fail)
    with pytest.raises(OSError, match="disk full"):
        writer.append(route)
    assert writer.failed
    assert json.loads((writer.directory / "status.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_buffer_limit(tmp_path, limit):
    root, route, _, _ = fixtures(tmp_path)
    with pytest.raises(IntegrityError):
        WorkerWriter(root, route.owner, limit)


def test_declared_input_content_mismatch_fails(tmp_path):
    root, route, peer, result = fixtures(tmp_path)
    writer = WorkerWriter(root, route.owner)
    writer.declare_execution("target", [replace(route.row, token_id=10)])
    writer.append(route)
    writer.append(result)
    writer.close()
    WorkerWriter(root, peer.observer).close()
    with pytest.raises(IntegrityError, match="declared input"):
        collect_run(root)


def test_declared_replica_missing_fails_even_when_worker_closed(tmp_path):
    root, route, peer, result = fixtures(tmp_path)
    writer = WorkerWriter(root, route.owner)
    writer.declare_execution("target", [route.row], [route.owner, peer.observer])
    writer.append(route)
    writer.append(result)
    writer.close()
    WorkerWriter(root, peer.observer).close()
    with pytest.raises(IntegrityError, match="replica observer"):
        collect_run(root)


def test_byte_bound_flushes_and_oversized_event_fails(tmp_path):
    root, route, peer, result = fixtures(tmp_path)
    writer = WorkerWriter(root, route.owner, shard_records=100, buffer_bytes=1000)
    writer.declare_execution("target", [route.row])
    writer.append(route)
    writer.append(result)
    assert writer.buffer_size <= 1000
    assert writer.shards
    writer.close()
    writer = WorkerWriter(root, peer.observer, buffer_bytes=1)
    with pytest.raises(IntegrityError, match="capacity"):
        writer.append(peer)
    assert writer.failed
