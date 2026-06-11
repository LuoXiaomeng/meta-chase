"""Policy abstraction for AFF / Meta-Chase (paper Section III)."""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple


# Motif statistics  (S)  -- updated after each execution; drives self-evolution

@dataclass
class MotifStats:
    runs:        int   = 0
    total_gain:  float = 0.0   # dAssoc observed: P_out facts contributed
    total_noise: float = 0.0   # low-confidence / conflicting derivations
    total_cost:  float = 0.0   # wall-clock seconds

    @property
    def avg_gain(self) -> float:
        return self.total_gain / self.runs if self.runs else 0.0

    @property
    def avg_noise(self) -> float:
        return self.total_noise / self.runs if self.runs else 0.0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.runs if self.runs else 0.0

    def update(self, gain: float, noise: float, cost: float) -> None:
        self.runs += 1
        self.total_gain += gain
        self.total_noise += noise
        self.total_cost += cost


# Task unit (SCC block) metadata  -- a motif's (phi, C, S) at block granularity

@dataclass
class TaskInfo:
    task_id:        int
    n_rules:        int
    mean_pca:       float
    total_support:  int
    produces_p_out: bool
    p_out_support:  int               = 0                      # support of P_out-producing rules
    downstream_p_out_support: int     = 0                      # self + DAG-reachable P_out support (static look-ahead)
    descendants:    FrozenSet[int]    = field(default_factory=frozenset)  # DAG-reachable block ids (for learned look-ahead)
    requires:       FrozenSet[str]    = field(default_factory=frozenset)  # applicability C
    stats:          MotifStats        = field(default_factory=MotifStats)  # S (updatable)


def build_task_infos(clusters_meta, p_out_preds, dag_edges=None) -> List[TaskInfo]:
    """Build TaskInfo list from blocks (clusters) of rule-dicts."""
    p_out = set(p_out_preds)
    infos: List[TaskInfo] = []
    for i, cluster in enumerate(clusters_meta):
        pcas = [r.get("pca", 0.0) for r in cluster]
        sup = sum(r.get("support", 0) for r in cluster)
        heads = {r["head_atom"][1] for r in cluster}
        pout_sup = sum(r.get("support", 0) for r in cluster
                       if r["head_atom"][1] in p_out)
        body: Set[str] = set()
        for r in cluster:
            for a in r["body_atoms"]:
                body.add(a[1])
        infos.append(TaskInfo(
            task_id=i,
            n_rules=len(cluster),
            mean_pca=sum(pcas) / len(pcas) if pcas else 0.0,
            total_support=sup,
            produces_p_out=bool(heads & p_out),
            p_out_support=pout_sup,
            requires=frozenset(body),
        ))

    if dag_edges:
        succ: Dict[int, Set[int]] = defaultdict(set)
        for a, b in dag_edges:
            succ[a].add(b)

        def reach(n: int) -> Set[int]:
            seen, stack = set(), [n]
            while stack:
                x = stack.pop()
                for m in succ.get(x, ()):
                    if m not in seen:
                        seen.add(m); stack.append(m)
            return seen
        n = len(infos)
        for info in infos:
            desc = reach(info.task_id)
            info.descendants = frozenset(d for d in desc if d < n)
            info.downstream_p_out_support = info.p_out_support + sum(
                infos[d].p_out_support for d in info.descendants)
    else:
        for info in infos:
            info.downstream_p_out_support = info.p_out_support
    return infos


def task_dag_from_meta(clusters_meta) -> Set[Tuple[int, int]]:
    """Producer->consumer DAG over an ARBITRARY task partition (B-policy)."""
    producers: Dict[str, Set[int]] = defaultdict(set)
    bodies: List[Set[str]] = []
    for i, c in enumerate(clusters_meta):
        b = {a[1] for r in c for a in r["body_atoms"]}
        bodies.append(b)
        for r in c:
            producers[r["head_atom"][1]].add(i)
    edges: Set[Tuple[int, int]] = set()
    for j, b in enumerate(bodies):
        for bp in b:
            for i in producers.get(bp, ()):
                if i != j:
                    edges.add((i, j))
    return edges


# Topological order with priority tie-break (the DAG-aware select+order, #5)

def topo_order(task_ids: Sequence[int],
               dag_edges: Set[Tuple[int, int]],
               priority: Dict[int, float]) -> List[int]:
    """Return a priority-driven topological-ish order of `task_ids`."""
    ids = list(task_ids)
    idset = set(ids)
    succ: Dict[int, List[int]] = {i: [] for i in ids}
    indeg: Dict[int, int] = {i: 0 for i in ids}
    for a, b in dag_edges:
        if a in idset and b in idset:
            succ[a].append(b)
            indeg[b] += 1
    # max-priority-first among ready (heapq is a min-heap -> negate)
    ready = [(-priority.get(i, 0.0), i) for i in ids if indeg[i] == 0]
    heapq.heapify(ready)
    placed: Set[int] = set()
    order: List[int] = []
    while len(placed) < len(ids):
        u = None
        while ready:
            _, cand = heapq.heappop(ready)
            if cand not in placed:
                u = cand
                break
        if u is None:                              # cycle: force highest-priority unplaced
            u = max((i for i in ids if i not in placed),
                    key=lambda i: priority.get(i, 0.0))
        placed.add(u)
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0 and v not in placed:
                heapq.heappush(ready, (-priority.get(v, 0.0), v))
    return order


# Policy interface

class Policy(ABC):
    name: str = "policy"

    @abstractmethod
    def priorities(self, tasks: List[TaskInfo]) -> Dict[int, float]:
        """task_id -> priority (higher = scheduled earlier)."""
        ...

    def order(self, tasks: List[TaskInfo],
              dag: Optional[Set[Tuple[int, int]]] = None) -> List[int]:
        pr = self.priorities(tasks)
        if dag:
            return topo_order([t.task_id for t in tasks], dag, pr)
        return sorted((t.task_id for t in tasks), key=lambda i: pr[i], reverse=True)


class FIFOPolicy(Policy):
    """FIFO: among DAG-ready tasks, pick the earliest one by task index (a fixed topological ordering)."""
    name = "fifo"

    def priorities(self, tasks):
        return {t.task_id: -float(t.task_id) for t in tasks}


class RewardGuidedPolicy(Policy):
    """AFF reward-guided policy."""
    name = "reward_guided"

    def __init__(self, lam: float = 1.0, mu: float = 0.5,
                 explore: float = 0.25, adaptive: bool = True,
                 nonpout_weight: float = 0.5, cold_start: bool = False,
                 lookahead: bool = True, down_weight: float = 1.0):
        self.lam = lam                  # noise weight
        self.mu = mu                    # cost weight
        self.explore = explore          # exploration bonus for never-run units
        self.adaptive = adaptive        # read S (True) or static-only (False, for ablation)
        self.nonpout_weight = nonpout_weight  # how much non-P_out support counts toward dAssoc
        self.cold_start = cold_start    # untried prior is NEUTRAL (uniform) -> learning curve
        self.lookahead = lookahead      # dAssoc credits DOWNSTREAM P_out (DAG look-ahead)
        # weight on the DEFERRED downstream credit vs IMMEDIATE own yield. <1 makes a
        self.down_weight = down_weight

    def priorities(self, tasks):
        max_sup = max((t.total_support for t in tasks), default=1) or 1
        max_rule = max((t.n_rules for t in tasks), default=1) or 1
        max_cost = max((t.stats.avg_cost for t in tasks), default=0.0)
        by_id = {t.task_id: t for t in tasks}

        def learned_gain(t):
            # observed P_out gain (from S) with DAG look-ahead: own yield + the
            g = t.stats.avg_gain
            if self.lookahead:
                g += self.down_weight * sum(by_id[d].stats.avg_gain
                                            for d in t.descendants if d in by_id)
            return g
        max_gain = max((learned_gain(t) for t in tasks), default=0.0)
        out: Dict[int, float] = {}
        for t in tasks:
            if self.adaptive and t.stats.runs > 0:
                # learned: dAssoc = observed (look-ahead) P_out contribution from S
                assoc = (learned_gain(t) / max_gain) if max_gain > 0 else 0.0
                noise = t.stats.avg_noise
                cost = (t.stats.avg_cost / max_cost) if max_cost > 0 else 0.0
                out[t.task_id] = assoc - self.lam * noise - self.mu * cost
            elif self.cold_start:
                # uninformed prior: every untried unit is equal (uniform) -> round-1
                out[t.task_id] = self.explore if t.stats.runs == 0 else 0.0
            else:
                # informed prior: dAssoc folds P_out. With look-ahead, an evidence
                pout = t.p_out_support + (
                    self.down_weight * (t.downstream_p_out_support - t.p_out_support)
                    if self.lookahead else 0)
                assoc = (pout
                         + self.nonpout_weight * (t.total_support - t.p_out_support)) / max_sup
                noise = 1.0 - t.mean_pca
                cost = t.n_rules / max_rule
                bonus = self.explore if t.stats.runs == 0 else 0.0   # exploration
                out[t.task_id] = assoc - self.lam * noise - self.mu * cost + bonus
        return out


