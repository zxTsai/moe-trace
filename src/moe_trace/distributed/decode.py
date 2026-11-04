"""CPU-testable Decode event contract. No device hooks or production collector.

The caller supplies verified input ownership; TP0 is not assumed to own all rows.
Speculation results annotate candidates, never erase their executed routes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from ..errors import IntegrityError


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise IntegrityError(f"{name} must be nonempty text")


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise IntegrityError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class Worker:
    deployment: str
    run: str
    node: str
    worker: str
    dp_rank: int
    tp_rank: int
    ep_rank: int
    device: int

    def __post_init__(self):
        for name in ("deployment", "run", "node", "worker"):
            _text(getattr(self, name), name)
        for name in ("dp_rank", "tp_rank", "ep_rank", "device"):
            _integer(getattr(self, name), name)


@dataclass(frozen=True)
class DecodeRow:
    source_engine: str
    execution: str
    scheduler_step: int
    global_request: str
    local_request: str
    input_row: int
    position: int
    token_id: int | None
    kind: str
    candidate_round: str | None = None
    candidate_node: str | None = None
    input_role: str = "sequence"

    def __post_init__(self):
        for name in ("source_engine", "execution", "global_request", "local_request"):
            _text(getattr(self, name), name)
        for name in ("scheduler_step", "input_row", "position"):
            _integer(getattr(self, name), name)
        if self.input_role not in {"sequence", "seed", "context", "mask", "redacted"}:
            raise IntegrityError("unsupported input role")
        if self.input_role == "redacted":
            if self.token_id is not None:
                raise IntegrityError("redacted token ID must be absent")
        elif self.input_role == "mask":
            if self.kind != "dspark_draft" or self.token_id is not None:
                raise IntegrityError("draft mask is not an ordinary vocabulary token")
        else:
            _integer(self.token_id, "token_id")
        if self.kind not in {"target_decode", "target_verify", "dspark_draft", "mtp_draft"}:
            raise IntegrityError("unsupported execution kind")
        if (self.candidate_round is None) != (self.candidate_node is None):
            raise IntegrityError("candidate round and node must appear together")
        if (self.kind == "target_verify" and self.candidate_node is None
                and self.input_role not in {"seed", "redacted"}):
            raise IntegrityError("verified candidate needs explicit identity")
        if self.kind == "target_decode" and self.candidate_node is not None:
            raise IntegrityError("ordinary decode cannot label a speculative candidate")
        if self.candidate_node is not None:
            _text(self.candidate_round, "candidate_round")
            _text(self.candidate_node, "candidate_node")


@dataclass(frozen=True)
class Route:
    observer: Worker
    owner: Worker
    row: DecodeRow
    model_scope: str
    layer: int
    num_experts: int
    expert_ids: tuple[int, ...]
    weights: tuple[float, ...] | None = None

    def __post_init__(self):
        if (self.observer.deployment, self.observer.run) != (self.owner.deployment, self.owner.run):
            raise IntegrityError("owner and observer belong to different runs")
        _text(self.model_scope, "model_scope")
        _integer(self.layer, "layer")
        _integer(self.num_experts, "num_experts")
        if (type(self.expert_ids) is not tuple or not self.expert_ids
                or len(set(self.expert_ids)) != len(self.expert_ids)):
            raise IntegrityError("expert IDs must be a nonempty unique tuple")
        for expert in self.expert_ids:
            _integer(expert, "expert ID")
            if expert >= self.num_experts:
                raise IntegrityError("expert ID out of range")
        if self.weights is not None:
            if type(self.weights) is not tuple or len(self.weights) != len(self.expert_ids):
                raise IntegrityError("weights must be an aligned tuple")
            if any(type(w) not in (int, float) or not math.isfinite(w) or w < 0
                   for w in self.weights):
                raise IntegrityError("invalid expert weight")

    @property
    def key(self):
        # Execution identifies a real input; never deduplicate by request position.
        return (self.owner.deployment, self.owner.run, self.row.source_engine,
                self.row.execution, self.row.input_row, self.model_scope, self.layer)


@dataclass(frozen=True)
class CandidateResult:
    global_request: str
    source_engine: str
    candidate_round: str
    candidate_node: str
    accepted: bool

    def __post_init__(self):
        for name in ("global_request", "source_engine", "candidate_round", "candidate_node"):
            _text(getattr(self, name), name)
        if type(self.accepted) is not bool:
            raise IntegrityError("accepted must be boolean")

    @property
    def key(self):
        return (self.global_request, self.source_engine, self.candidate_round, self.candidate_node)


def merge_routes(routes):
    """Validate replicas and count each owner's row once. Small offline fixtures only."""
    routes = list(routes)
    runs = {(r.owner.deployment, r.owner.run) for r in routes}
    if len(runs) > 1:
        raise IntegrityError("cannot merge different deployments/runs")
    groups = {}
    for route in routes:
        groups.setdefault(route.key, []).append(route)
    merged = []
    for group in groups.values():
        baseline = group[0]
        seen = set()
        owners = []
        for route in group:
            if route.observer in seen:
                raise IntegrityError("duplicate observer record")
            seen.add(route.observer)
            if (route.owner, route.row, route.num_experts, route.expert_ids, route.weights) != (
                    baseline.owner, baseline.row, baseline.num_experts,
                    baseline.expert_ids, baseline.weights):
                raise IntegrityError("replica contents or ownership disagree")
            if route.observer == route.owner:
                owners.append(route)
        if len(owners) != 1:
            raise IntegrityError("missing unique owner record")
        merged.append(owners[0])
    return sorted(merged, key=lambda r: r.key)


def annotate_candidates(routes, results):
    """Join target-verification results; None means no result, not rejection.

    Draft rows need not identify the candidate they generate: these are input
    routes, so the producer must supply an explicit association if one exists.
    """
    routes = list(routes)
    if len({(r.owner.deployment, r.owner.run) for r in routes}) > 1:
        raise IntegrityError("candidate join requires a single run")
    result_map = {}
    for result in results:
        if result.key in result_map:
            raise IntegrityError("duplicate candidate result")
        result_map[result.key] = result.accepted
    known = {(r.row.global_request, r.row.source_engine,
              r.row.candidate_round, r.row.candidate_node) for r in routes
             if r.row.kind == "target_verify" and r.row.candidate_node is not None}
    if set(result_map) - known:
        raise IntegrityError("candidate result has no target verification route")
    return [(r, result_map.get((r.row.global_request, r.row.source_engine,
                              r.row.candidate_round, r.row.candidate_node))) for r in routes]


def expert_counts(routes):
    """Executed logical choices, including rejected candidates; no physical load claim."""
    return Counter((r.row.kind, r.model_scope, r.layer, expert)
                   for r in merge_routes(routes) for expert in r.expert_ids)
