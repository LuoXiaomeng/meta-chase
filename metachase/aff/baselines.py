"""baselines — the two deduction baselines AFF is compared against (paper Sec VI)."""

from __future__ import annotations

import time
from typing import List

from .. import confluence
from ..chase.coordinator import VanillaCoordinator
from ..graph.derived import DerivedFacts
from .config import AFFConfig
from .runner import AFFResult, AFFRunner


def run_ff(ctx, record_events: bool = True, max_rounds: int = 0,
           canonicalize: bool = True) -> AFFResult:
    """FF baseline: a single flat chase of the focused GAR set Σ to closure."""
    return AFFRunner(ctx, AFFConfig(
        task_source="all_rules_focused", order_strategy="natural",
        sharing="shared", canonicalize=canonicalize,
        record_p_out_events=record_events, max_rounds=max_rounds,
    )).run()


def run_ff_task(ctx, n_tasks: int, record_events: bool = True,
                max_rounds: int = 0, scope: str = "focused",
                mode: str = "canon", gars=None) -> AFFResult:
    """FF_task baseline: re-run the ENTIRE rule set FRESH for each of N task slots."""
    if gars is None:
        gars = ctx.gars_focused if scope == "focused" else ctx.gars_full
    early_commit = (mode == "EC")
    t0 = time.perf_counter()
    result = AFFResult(p_out=set())
    per_task_timing: List[float] = []
    cum_steps = 0       # global chasing-step offset across the N fresh slots
    for k in range(max(1, n_tasks)):
        ts = time.perf_counter()
        coord = VanillaCoordinator(
            ctx.graph, gars, max_rounds=max_rounds, max_matches=0, verbose=False,
            matcher=ctx.matcher, p_out_ids=ctx.p_out_ids,
            functional_p_out=ctx.functional_p_out_ids,
            early_commit=early_commit, record_p_out_events=record_events)
        res = coord.run(existing_derived=DerivedFacts(), existing_fired_pairs=set(),
                        task_id=f"FFT{k}")
        per_task_timing.append(time.perf_counter() - ts)
        result.all_results.extend(res)
        result.per_task.append((k, coord.run_stats))
        for triple, gids in coord.p_out_provenance.items():
            result.provenance.setdefault(triple, set()).update(gids)
        if record_events:
            result.p_out_event_log.extend(
                (s, l, o, g, n, t - t0, cum_steps + step)
                for (s, l, o, g, n, t, step) in coord.p_out_events)
        cum_steps += coord.run_stats.chasing_steps
    if early_commit:
        result.p_out = confluence.collect_p_out(result.all_results, ctx.p_out_ids)
    else:
        result.p_out = confluence.canonicalize(
            result.all_results, ctx.p_out_ids, ctx.pca_by_gar,
            functional=ctx.functional_p_out, provenance=result.provenance)
    result.timing["total"] = time.perf_counter() - t0
    result.timing["per_task"] = per_task_timing
    return result
