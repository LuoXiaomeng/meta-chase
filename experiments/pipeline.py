"""pipeline — the single flow turning a Method into a uniform `Trace` via one engine call, so
every arm (baselines included) takes the same path and metrics/plotting never branch on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Set, Tuple

from .spec import Method

# decision.kind -> the run_ff_task `mode` string
_FFTASK_MODE = {"canon": "canon", "ec": "EC", "two_phase": "two_phase"}
# decision.kind -> the run_aff_unified `mode` string
_UNIFIED_MODE = {"canon": "canon", "monotone": "canon", "ec": "EC",
                 "two_phase": "two-phase"}


@dataclass
class Trace:
    """A backend-uniform view of one arm's run. Metrics read only these fields."""
    name: str
    p_out: Set[Tuple]
    result: Any                      # the underlying AFFResult (for rich metrics)
    total_steps: int                 # authoritative total chasing steps
    invocations: int                 # rule invocations (generated events)
    runtime: float                   # wall-clock seconds (timing['total'])
    n_tasks: int
    event_log: List = field(default_factory=list)   # P_out arrival events


def _total_steps(result) -> int:
    return sum(rs.chasing_steps for _ci, rs in result.per_task)


def _invocations(result) -> int:
    return sum(rs.gen_events for _ci, rs in result.per_task)


def run(ctx, method: Method) -> Trace:
    """Execute one Method against a loaded Context and return its Trace."""
    rp = method.partition.build(ctx)

    if method.runner == "ff_task":
        from metachase import run_ff_task
        mode = _FFTASK_MODE[method.decision.kind]
        slots = method.slots if method.slots is not None else rp.n_tasks
        # re-run the partition's own rule set each slot (flattened from its tasks),
        # falling back to the scope default when the partition carries no explicit gars.
        gars = [g for t in rp.tasks for g in t["gars"]] if rp.tasks else None
        result = run_ff_task(ctx, slots, record_events=method.record_events,
                             max_rounds=method.max_rounds, scope=method.partition.scope,
                             mode=mode, gars=gars)
        return _trace(method.name, result, ctx)

    if method.runner == "aff_unified":
        # the structure-adaptive unified policy (sparse SCC -> budget-greedy static
        # order; dense -> dynamic readiness). Order is computed from startup history.
        from . import scheduling
        from .experiment import _rule_stats_for
        mode = _UNIFIED_MODE[method.decision.kind]
        tasks = None if method.partition.kind == "scc_blocks" else rp.tasks
        result, _info = scheduling.run_aff_unified(
            ctx, _rule_stats_for(ctx), tasks=tasks, mode=mode,
            record_events=method.record_events)
        return _trace(method.name, result, ctx)

    # 'aff' and 'ff' both run through AFFRunner; 'ff' is just sugar for an
    # all-rules partition (a flat single-task chase).
    from metachase import AFFConfig, AFFRunner
    cfg_kwargs = dict(
        sharing="shared", seed=method.seed, max_rounds=method.max_rounds,
        record_p_out_events=method.record_events,
        shuffle_rules_seed=method.shuffle_seed,
        to_quiescence=method.to_quiescence, max_passes=method.max_passes,
    )
    cfg_kwargs.update(rp.config)
    cfg_kwargs.update(method.schedule.build(ctx, rp))
    cfg_kwargs.update(method.decision.build())
    result = AFFRunner(ctx, AFFConfig(**cfg_kwargs)).run()
    return _trace(method.name, result, ctx)


def _trace(name, result, ctx) -> Trace:
    return Trace(
        name=name,
        p_out=set(result.p_out),
        result=result,
        total_steps=_total_steps(result),
        invocations=_invocations(result),
        runtime=float(result.timing.get("total", 0.0)),
        n_tasks=len(result.per_task),
        event_log=list(getattr(result, "p_out_event_log", []) or []),
    )
