"""scheduling — explicit policy-order computation for arms that need a probe run or startup
statistics: `budget_greedy` and the unified structure-adaptive AFF policy (`aff_unified`).
"""

from __future__ import annotations

import collections
import random
from typing import Dict, List


def _mode_kwargs(mode: str) -> dict:
    if mode == "EC":
        return dict(decision_mode="early_commit", canonicalize=False)
    if mode == "canon":
        return dict(decision_mode="monotone", canonicalize=True)
    if mode == "two-phase":
        return dict(decision_mode="two_phase", two_phase_scope="task")
    raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Historical statistics (task-independent warmups)
# ---------------------------------------------------------------------------
def rule_stats_canonical(ctx, max_passes: int = 8) -> dict:
    """One canonical task-independent warmup: chase the focused SCC blocks in a fixed
    topological order to closure and collect per-rule (steps, canon-wins)."""
    from metachase import AFFConfig, AFFRunner
    afr = AFFRunner(ctx, AFFConfig(
        task_source="scc_blocks", order_strategy="natural", sharing="shared",
        canonicalize=True, to_quiescence=True, max_passes=max_passes)).run()
    steps: Dict[str, int] = {}
    for r in afr.all_results:
        steps[r.gar_id] = steps.get(r.gar_id, 0) + 1
    yld: Dict[str, float] = {}
    for triple in afr.p_out:
        producers = afr.provenance.get(triple, set())
        if producers:
            winner = max(producers, key=lambda g: ctx.pca_by_gar.get(g, 0.0))
            yld[winner] = yld.get(winner, 0.0) + 1.0
    return {"steps": steps, "yield": yld}


def topo_warmup_stats(ctx, max_passes: int = 8):
    """Topo-cumulative warmup over SCC blocks: per-block (steps, marginal canon
    yield). Order-independent because each block is measured with its ancestors
    present (deps-first cumulative pass)."""
    from metachase import AFFConfig, AFFRunner
    probe = AFFRunner(ctx, AFFConfig(
        task_source="scc_blocks", order_strategy="natural", sharing="shared",
        canonicalize=True, track_snapshots=True, to_quiescence=True,
        max_passes=max_passes)).run()
    steps: Dict[int, int] = {}
    yld: Dict[int, float] = {}
    prev = 0
    for k, (ci, rs) in enumerate(probe.per_task):
        s = probe.per_task_snapshots[k] if k < len(probe.per_task_snapshots) else {}
        cur = len(s.get("cumulative_canonical", s.get("cumulative_p_out", [])))
        yld[ci] = yld.get(ci, 0.0) + max(cur - prev, 0)
        prev = cur
        steps[ci] = steps.get(ci, 0) + rs.chasing_steps
    return steps, yld


def task_stats_from_rules(tasks, rule_stats):
    """Aggregate per-rule historical stats to per-task (steps, yield) — pure lookup."""
    rs, ry = rule_stats["steps"], rule_stats["yield"]
    steps: Dict[int, int] = {}
    yld: Dict[int, float] = {}
    for i, t in enumerate(tasks):
        gars = t["gars"] if isinstance(t, dict) else t
        steps[i] = sum(rs.get(g.id, 0) for g in gars)
        yld[i] = sum(ry.get(g.id, 0.0) for g in gars)
    return steps, yld


def unified_prior(ctx, tasks, rule_stats, lam: float = 1.0, eps: float = 1.0):
    """Per-task prior = value-density x noise-factor: (yield/cost)*(1-lam*noise)."""
    steps, yld = task_stats_from_rules(tasks, rule_stats)
    prior = []
    for i, t in enumerate(tasks):
        gars = t["gars"] if isinstance(t, dict) else t
        pcas = [ctx.pca_by_gar.get(g.id, 0.0) for g in gars]
        mean_pca = (sum(pcas) / len(pcas)) if pcas else 0.0
        density = yld.get(i, 0.0) / max(steps.get(i, 0), eps)
        prior.append(density * max(0.0, 1.0 - lam * (1.0 - mean_pca)))
    return prior


# ---------------------------------------------------------------------------
# Order computation
# ---------------------------------------------------------------------------
def budget_greedy_order(steps: Dict[int, int], yld: Dict[int, float],
                        dag, blocks) -> List[int]:
    """DAG-aware budget-optimal greedy order: repeatedly pick the block whose
    addition (it + its not-yet-selected ancestor closure) gives the best
    Δyield/Δsteps density; append that closure ancestor-first."""
    preds: Dict[int, set] = collections.defaultdict(set)
    for a, b in dag:
        preds[b].add(a)

    def ancestors(i: int) -> set:
        seen, st = set(), [i]
        while st:
            x = st.pop()
            for p in preds.get(x, ()):
                if p not in seen:
                    seen.add(p); st.append(p)
        return seen

    anc = {i: ancestors(i) for i in blocks}
    order: List[int] = []
    sel: set = set()
    remaining = set(blocks)
    while remaining:
        best_i, best_d, best_clo = None, None, None
        for i in remaining:
            clo = (anc[i] | {i}) - sel
            cost = sum(steps.get(j, 0) for j in clo)
            gain = sum(yld.get(j, 0.0) for j in clo)
            d = gain / cost if cost > 0 else (gain if gain > 0 else 0.0)
            if best_d is None or d > best_d:
                best_d, best_i, best_clo = d, i, clo
        for j in sorted(best_clo):
            order.append(j); sel.add(j); remaining.discard(j)
    return order


def random_topo_order(n: int, dag, seed: int = 42) -> List[int]:
    """A seeded random topological linearisation of the task-DAG (the schedule-quality
    baseline). Picks a random ready task at each step."""
    preds: Dict[int, set] = collections.defaultdict(set)
    for a, b in dag:
        preds[b].add(a)
    rng = random.Random(seed)
    placed: set = set()
    order: List[int] = []
    while len(order) < n:
        ready = [i for i in range(n) if i not in placed and preds[i] <= placed]
        if not ready:                       # cycle / unreachable: take any remaining
            ready = [i for i in range(n) if i not in placed]
        i = rng.choice(ready)
        order.append(i); placed.add(i)
    return order


def warmed_policy_order(ctx, tasks, dag=None, seed: int = 42, max_passes: int = 12):
    """The reward policy WARMED from a probe: run the tasks once (natural order, to
    quiescence) measuring each task's actual canon P_out gain + cost, fill the policy
    stats, then return the reward order computed with that learned history."""
    from metachase import AFFConfig, AFFRunner
    from metachase.policy import (task_dag_from_meta, build_task_infos,
                                  RewardGuidedPolicy)
    probe = AFFRunner(ctx, AFFConfig(
        task_source="prebuilt", prebuilt_tasks=tasks, order_strategy="natural",
        sharing="shared", canonicalize=True, track_snapshots=True,
        to_quiescence=True, max_passes=max_passes, seed=seed)).run()
    per_t = probe.timing.get("per_task", [])
    snaps = probe.per_task_snapshots
    gain: Dict[int, float] = {}
    cost: Dict[int, float] = {}
    prev = 0
    for k, (ci, _rs) in enumerate(probe.per_task):
        s = snaps[k] if k < len(snaps) else {}
        cur = len(s.get("cumulative_canonical", s.get("cumulative_p_out", [])))
        gain[ci] = gain.get(ci, 0.0) + max(cur - prev, 0)
        prev = cur
        cost[ci] = cost.get(ci, 0.0) + (per_t[k] if k < len(per_t) else 0.0)
    mbi = {m["id"]: m for m in ctx.meta_full}
    cm = [[mbi[g.id] for g in (t["gars"] if isinstance(t, dict) else t)
           if g.id in mbi] for t in tasks]
    if dag is None:
        dag = task_dag_from_meta(cm)
    infos = build_task_infos(cm, ctx.cfg.p_out, dag_edges=dag)
    for info in infos:
        info.stats.update(gain=float(gain.get(info.task_id, 0.0)), noise=0.0,
                          cost=float(cost.get(info.task_id, 0.0)))
    return RewardGuidedPolicy(adaptive=True, lookahead=True).order(infos, dag=dag)


def random_topo_orders(ctx, n_schedules: int, seed: int, k=None) -> List[List[int]]:
    """n random topological linearisations of the SCC-block DAG (priority-based: a
    seeded random priority per block, then a topo sort). Matches the engine's
    `_random_topo_orders` RNG sequence exactly, so random-schedule curves reproduce."""
    from metachase.policy import topo_order
    rng = random.Random(seed)
    ids = list(range(ctx.n_tasks if k is None else k))
    orders = []
    for _ in range(n_schedules):
        pr = {i: rng.random() for i in ids}
        orders.append(topo_order(ids, ctx.block_dag, pr))
    return orders


def resolve_order(kind: str, ctx, rp, seed: int = 42) -> List[int]:
    """Compute an explicit task order for a policy-based Schedule (spec.Schedule)."""
    if kind == "random":
        return random_topo_order(rp.n_tasks, rp.dag, seed)
    if kind in ("budget_greedy", "reward"):
        # SCC blocks: per-block MEASURED marginal (best); else rule-aggregated.
        from .experiment import _rule_stats_for
        is_scc = rp.config.get("task_source") == "scc_blocks"
        if is_scc:
            steps, yld = topo_warmup_stats(ctx)
        else:
            steps, yld = task_stats_from_rules(rp.tasks, _rule_stats_for(ctx))
        return budget_greedy_order(steps, yld, rp.dag, list(range(rp.n_tasks)))
    raise ValueError(f"no order resolver for schedule {kind!r}")


# ---------------------------------------------------------------------------
# The unified policy (exposed as the `aff_unified` runner)
# ---------------------------------------------------------------------------
def run_aff_unified(ctx, rule_stats, tasks=None, mode: str = "canon",
                    record_events: bool = True, density_tau: float = 0.2,
                    max_passes: int = 10):
    """Structure-adaptive UNIFIED AFF policy. Picks the execution model from the
    task-DAG density: sparse (structured SCC blocks) -> static value-density
    (budget-greedy) order; dense (random partition) -> dynamic Γ-aware readiness with
    the value-density x noise prior. tasks=None => SCC blocks. Returns (AFFResult,
    info={'density','branch'})."""
    from metachase import AFFConfig, AFFRunner
    from metachase.policy import task_dag_from_meta
    scc = tasks is None
    if scc:
        task_list = [{"gars": list(c)} for c in ctx.clusters_gars]
        dag = ctx.block_dag
    else:
        task_list = tasks
        mbi = {m["id"]: m for m in ctx.meta_full}
        cm = [[mbi[g.id] for g in (t["gars"] if isinstance(t, dict) else t)
               if g.id in mbi] for t in task_list]
        dag = task_dag_from_meta(cm)
    n = len(task_list)
    density = (len(dag) / (n * (n - 1) / 2)) if n > 1 else 0.0
    mk = _mode_kwargs(mode)
    src = (dict(task_source="scc_blocks") if scc
           else dict(task_source="prebuilt", prebuilt_tasks=task_list))
    if density < density_tau:                          # structured -> static order
        steps, yld = (topo_warmup_stats(ctx) if scc
                      else task_stats_from_rules(task_list, rule_stats))
        order = budget_greedy_order(steps, yld, dag, list(range(n)))
        extra = dict(order_strategy="custom", custom_order=order,
                     to_quiescence=True, max_passes=max_passes)
        branch = "static"
    else:                                              # dense/random -> dynamic
        extra = dict(order_strategy="dynamic_ready",
                     dynamic_yield_prior=unified_prior(ctx, task_list, rule_stats))
        branch = "dynamic"
    afr = AFFRunner(ctx, AFFConfig(sharing="shared",
        record_p_out_events=record_events, **src, **extra, **mk)).run()
    return afr, {"density": round(density, 4), "branch": branch}
