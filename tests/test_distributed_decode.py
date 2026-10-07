from dataclasses import replace

import pytest

from moe_trace.distributed.decode import (
    Worker, DecodeRow, Route, CandidateResult, merge_routes,
    annotate_candidates, expert_counts,
)
from moe_trace.errors import IntegrityError


def fixture():
    owner = Worker("deployment", "run", "D0", "worker0", 0, 0, 0, 0)
    peer = replace(owner, worker="worker1", tp_rank=1, ep_rank=1, device=1)
    row = DecodeRow("engine0", "exec1", 3, "requestA", "localA", 0, 100, 42,
                    "target_verify", "round1", "candidate0")
    route = Route(owner, owner, row, "target", 20, 4, (3, 1), (0.75, 0.25))
    return route, replace(route, observer=peer)


def test_reordered_replicas_count_once():
    route, replica = fixture()
    assert merge_routes([replica, route]) == [route]
    assert dict(expert_counts([replica, route])) == {
        ("target_verify", "target", 20, 3): 1,
        ("target_verify", "target", 20, 1): 1,
    }


def test_owner_can_be_nonzero_tp():
    route, replica = fixture()
    owner = replica.observer
    records = [replace(route, owner=owner), replace(replica, owner=owner)]
    assert merge_routes(records) == [records[1]]


@pytest.mark.parametrize("field,value", [
    ("expert_ids", (1, 3)), ("weights", (0.5, 0.5)),
    ("row", replace(fixture()[0].row, token_id=43)),
    ("owner", replace(fixture()[0].owner, worker="wrong")),
])
def test_replica_disagreement(field, value):
    route, replica = fixture()
    with pytest.raises(IntegrityError, match="disagree"):
        merge_routes([route, replace(replica, **{field: value})])


def test_missing_owner_and_duplicate_observer():
    route, replica = fixture()
    with pytest.raises(IntegrityError, match="owner"):
        merge_routes([replica])
    with pytest.raises(IntegrityError, match="duplicate"):
        merge_routes([route, route])


def test_recomputation_and_same_position_candidates_are_retained():
    route, _ = fixture()
    candidate = replace(route, row=replace(route.row, input_row=1, candidate_node="candidate1"))
    recompute = replace(route, row=replace(route.row, execution="exec2"))
    assert len(merge_routes([candidate, route, recompute])) == 3
    assert sum(expert_counts([candidate, route, recompute]).values()) == 6


def test_engine_namespaces_and_draft_layers_do_not_collide():
    route, _ = fixture()
    other = replace(route, row=replace(route.row, source_engine="engine1"))
    draft = replace(route, model_scope="draft",
                    row=replace(route.row, kind="dspark_draft"))
    assert len(merge_routes([other, draft, route])) == 3


def test_rejection_never_deletes_computed_routes():
    route, _ = fixture()
    result = CandidateResult("requestA", "engine0", "round1", "candidate0", False)
    assert annotate_candidates([route], [result]) == [(route, False)]
    assert sum(expert_counts([route]).values()) == 2
    assert annotate_candidates([route], []) == [(route, None)]


def test_unknown_and_duplicate_result_fail():
    route, _ = fixture()
    result = CandidateResult("requestA", "engine0", "round1", "candidate0", True)
    with pytest.raises(IntegrityError, match="duplicate"):
        annotate_candidates([route], [result, result])
    with pytest.raises(IntegrityError, match="no target"):
        annotate_candidates([route], [replace(result, source_engine="wrong")])


def test_cross_run_merge_and_join_fail():
    route, _ = fixture()
    worker = replace(route.owner, run="other")
    other = replace(route, owner=worker, observer=worker)
    for function in (lambda: merge_routes([route, other]),
                     lambda: annotate_candidates([route, other], [])):
        with pytest.raises(IntegrityError):
            function()


@pytest.mark.parametrize("changes", [
    {"expert_ids": (4, 1)}, {"expert_ids": (1, 1)},
    {"expert_ids": (True, 1)}, {"expert_ids": [3, 1]},
    {"weights": (float("nan"), 0.2)}, {"weights": (-0.1, 0.2)},
    {"weights": (True, 0.2)}, {"layer": -1},
])
def test_invalid_route_rejected(changes):
    with pytest.raises(IntegrityError):
        replace(fixture()[0], **changes)


@pytest.mark.parametrize("changes", [
    {"position": -1}, {"input_row": True}, {"kind": "prefill"},
    {"candidate_node": None}, {"candidate_round": ""}, {"global_request": ""},
])
def test_invalid_input_identity_rejected(changes):
    with pytest.raises(IntegrityError):
        replace(fixture()[0].row, **changes)


def test_draft_input_need_not_have_output_candidate_identity():
    row = replace(fixture()[0].row, kind="dspark_draft",
                  candidate_round=None, candidate_node=None)
    assert row.candidate_node is None
