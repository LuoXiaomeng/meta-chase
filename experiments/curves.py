"""curves — post-hoc reconstruction of the Exp-1 metrics (recall / overturns / coverage
vs chasing steps) from a run's ordered P_out event log. Nothing here runs on the chase.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

from metachase.confluence.canon import decide

Triple = Tuple[int, int, int]
Event = Tuple[int, int, int, str, bool, float, int]


# -- grids ----------------------------------------------------------------


def step_targets(total_steps: int, interval: int) -> List[int]:
    """Checkpoint step counts: multiples of `interval` up to `total_steps`, plus the
    final total so the last point is always the complete closure."""
    if interval <= 0:
        return [total_steps]
    ts = list(range(interval, total_steps + 1, interval))
    if not ts or ts[-1] != total_steps:
        ts.append(total_steps)
    return ts


# -- prefix selection -----------------------------------------------------
def _prefix_by_step(event_log: Sequence[Event], step: int) -> List[Event]:
    out: List[Event] = []
    for e in event_log:
        if e[6] <= step:
            out.append(e)
        else:
            break
    return out


def _prefix_by_time(event_log: Sequence[Event], t: float) -> List[Event]:
    out: List[Event] = []
    for e in event_log:
        if e[5] <= t:
            out.append(e)
        else:
            break
    return out


def time_at_step(event_log: Sequence[Event], targets: Sequence[int]) -> Dict[int, float]:
    """For each target total-step count, the t_rel of the last event with
    step_idx <= target (zero-order hold)."""
    out: Dict[int, float] = {}
    want = sorted(set(targets))
    i = 0
    last_t = 0.0
    for e in event_log:
        while i < len(want) and e[6] > want[i]:
            out[want[i]] = last_t
            i += 1
        if i >= len(want):
            break
        last_t = e[5]
    final_t = event_log[-1][5] if event_log else 0.0
    for w in want:
        out.setdefault(w, final_t)
    return out


# -- published set + precision/recall -------------------------------------
def published_set(prefix: Sequence[Event], pca_by_gar: Dict[str, float],
                  functional=None) -> Set[Triple]:
    """Resolve the prefix's P_out candidates to the published triple set (max PCA over
    all producers per triple -> order-independent winner, matching canonicalize)."""
    weight: Dict[Triple, float] = {}
    cand_objs: "defaultdict[Tuple[int,int], set]" = defaultdict(set)
    for (s, l, o, g, is_new, _t, _step) in prefix:
        w = pca_by_gar.get(g, 0.0)
        k3 = (s, l, o)
        if w > weight.get(k3, -1.0):
            weight[k3] = w
        if is_new:
            cand_objs[(s, l)].add(o)
    candidates = {(s, l): [(o, weight[(s, l, o)]) for o in objs]
                  for (s, l), objs in cand_objs.items()}
    return decide(candidates, functional)


def precision_recall(published: Set[Triple], gt: Set[Triple]) -> Tuple[float, float]:
    if not gt:
        return 0.0, 0.0
    tp = len(published & gt)
    p = tp / len(published) if published else 0.0
    return p, tp / len(gt)


def accuracy_curve(event_log: Sequence[Event], gt: Set[Triple],
                   pca_by_gar: Dict[str, float], functional,
                   step_axis: Sequence[int]) -> List[Dict]:
    """Exp-1.1: recall (vs canon GT) at each chasing-step checkpoint —
    recall = correct P_out / |GT|."""
    tmap = time_at_step(event_log, step_axis)
    out: List[Dict] = []
    for s in step_axis:
        pub = published_set(_prefix_by_step(event_log, s), pca_by_gar, functional)
        tp = len(pub & gt)
        out.append({"step": s, "time": round(tmap.get(s, 0.0), 6),
                    "recall": round(tp / len(gt), 6) if gt else 0.0})
    return out


def coverage_vs_time(event_log: Sequence[Event], gt: Set[Triple],
                     pca_by_gar: Dict[str, float], functional,
                     time_grid: Sequence[float]) -> List[Dict]:
    """Exp-1.3: published recall/precision at each wall-clock checkpoint (anytime
    delivery under a TIME budget)."""
    out: List[Dict] = []
    for tg in time_grid:
        pub = published_set(_prefix_by_time(event_log, tg), pca_by_gar, functional)
        p, r = precision_recall(pub, gt)
        out.append({"time": round(tg, 6), "recall": round(r, 6),
                    "precision": round(p, 6), "n_published": len(pub)})
    return out


def _is_func(label, functional) -> bool:
    if functional is None:
        return True
    if isinstance(functional, dict):
        return functional.get(label, True)
    return label in functional


def overturn_curve(event_log: Sequence[Event], pca_by_gar: Dict[str, float],
                   functional, step_axis: Sequence[int]) -> List[Dict]:
    """Exp-1.2: cumulative result OVERTURNS under a running task-internal canon. An
    overturn = a functional key's max-PCA winner CHANGES when a higher-PCA competitor
    arrives later. The first value for a key is not an overturn."""
    keyvals: Dict[Tuple[int, int], Dict[int, float]] = {}
    winner: Dict[Tuple[int, int], int] = {}
    events: List[Tuple[int, int]] = [(0, 0)]
    cum = 0
    for (s, l, o, g, _is_new, _t, step) in event_log:
        if not _is_func(l, functional):
            continue
        w = pca_by_gar.get(g, 0.0)
        d = keyvals.setdefault((s, l), {})
        if w > d.get(o, -1.0):
            d[o] = w
        new_w = min(d.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if (s, l) in winner and new_w != winner[(s, l)]:
            cum += 1
            events.append((step, cum))
        winner[(s, l)] = new_w
    out: List[Dict] = []
    j, last = 0, 0
    for s in step_axis:
        while j < len(events) and events[j][0] <= s:
            last = events[j][1]; j += 1
        out.append({"step": s, "overturns": last})
    return out
