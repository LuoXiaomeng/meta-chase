"""spec — the declarative vocabulary of an arm: Partition × Schedule × Decision × Method, each
resolving to a slice of `AFFConfig` kwargs. Pure descriptors; the running lives in `pipeline.run`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Partition — how the rule set becomes tasks
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedPartition:
    """A built partition: the AFFConfig task kwargs + the materialised task list and
    task-DAG (needed by policy-based schedules)."""
    config: dict
    tasks: List          # list of task dicts ({'gars': [...]}) or cluster gar-lists
    dag: set             # task-DAG edges {(producer_idx, consumer_idx)}
    n_tasks: int


@dataclass(frozen=True)
class Partition:
    """How Σ is split into tasks. Use the classmethods, not the raw constructor."""
    kind: str                       # scc_blocks | merged_scc | random | all_rules | prebuilt
    n: Optional[int] = None
    seed: int = 42
    scope: str = "focused"          # focused | full   (random / all_rules)
    tasks: Optional[List] = None    # prebuilt only

    @classmethod
    def scc_blocks(cls) -> "Partition":
        """Σ's native strongly-connected dependency blocks (engine-built)."""
        return cls("scc_blocks")

    @classmethod
    def merged_scc(cls, n: int) -> "Partition":
        """SCC blocks merged into exactly n contiguous-topological buckets."""
        return cls("merged_scc", n=n)

    @classmethod
    def random(cls, n: int, seed: int = 42, scope: str = "focused") -> "Partition":
        """Σ shuffled (seeded) into n round-robin buckets."""
        return cls("random", n=n, seed=seed, scope=scope)

    @classmethod
    def all_rules(cls, focused: bool = True) -> "Partition":
        """A single task holding all of Σ (the FF baseline's flat chase)."""
        return cls("all_rules", scope="focused" if focused else "full")

    @classmethod
    def prebuilt(cls, tasks: List) -> "Partition":
        """A caller-supplied task list."""
        return cls("prebuilt", tasks=list(tasks))

    # -- resolution -------------------------------------------------------
    def build(self, ctx) -> ResolvedPartition:
        from metachase import build_merged_scc_tasks, build_random_partition_tasks
        if self.kind == "scc_blocks":
            tasks = [{"gars": list(c)} for c in ctx.clusters_gars]
            return ResolvedPartition({"task_source": "scc_blocks"}, tasks,
                                     set(ctx.block_dag), len(tasks))
        if self.kind == "all_rules":
            src = "all_rules_focused" if self.scope == "focused" else "all_rules"
            gars = ctx.gars_focused if self.scope == "focused" else ctx.gars_full
            return ResolvedPartition({"task_source": src},
                                     [{"gars": list(gars)}], set(), 1)
        if self.kind == "merged_scc":
            tasks = build_merged_scc_tasks(ctx, self.n)
        elif self.kind == "random":
            tasks = build_random_partition_tasks(ctx, self.n, self.seed, self.scope)
        elif self.kind == "prebuilt":
            tasks = self.tasks
        else:
            raise ValueError(f"unknown partition kind {self.kind!r}")
        return ResolvedPartition(
            {"task_source": "prebuilt", "prebuilt_tasks": tasks},
            tasks, _task_dag(ctx, tasks), len(tasks))


def _task_dag(ctx, tasks) -> set:
    """The task-DAG over a prebuilt task list, derived from rule metadata."""
    from metachase.policy import task_dag_from_meta
    mbi = {m["id"]: m for m in ctx.meta_full}
    cm = [[mbi[g.id] for g in (t["gars"] if isinstance(t, dict) else t)
           if g.id in mbi] for t in tasks]
    return task_dag_from_meta(cm)


# ---------------------------------------------------------------------------
# Schedule — in what order tasks run (random IS just a schedule)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Schedule:
    """Task ordering policy. Engine-native strategies resolve to a strategy string;
    policy-based ones (reward / budget_greedy / random) compute an explicit order and
    feed it as `custom_order`."""
    kind: str                       # natural | fifo | dynamic | custom | random | reward | budget_greedy
    seed: int = 42
    order: Optional[List[int]] = None        # custom only
    yield_prior: Optional[List[float]] = None  # dynamic only

    @classmethod
    def natural(cls) -> "Schedule":
        return cls("natural")

    @classmethod
    def fifo(cls) -> "Schedule":
        return cls("fifo")

    @classmethod
    def dynamic(cls, yield_prior: Optional[List[float]] = None) -> "Schedule":
        """Γ-aware online readiness (good for dense / random partitions)."""
        return cls("dynamic", yield_prior=yield_prior)

    @classmethod
    def custom(cls, order: List[int]) -> "Schedule":
        return cls("custom", order=list(order))

    @classmethod
    def random(cls, seed: int = 42) -> "Schedule":
        """A random task order (the schedule-quality baseline). Realised as a seeded
        random topological linearisation of the task-DAG."""
        return cls("random", seed=seed)

    @classmethod
    def reward(cls, seed: int = 42) -> "Schedule":
        """The paper's reward-guided policy, warmed from a probe run."""
        return cls("reward", seed=seed)

    @classmethod
    def budget_greedy(cls) -> "Schedule":
        """Static value-density order (budget-greedy on measured yield/steps)."""
        return cls("budget_greedy")

    # -- resolution -------------------------------------------------------
    def build(self, ctx, rp: ResolvedPartition) -> dict:
        if self.kind == "natural":
            return {"order_strategy": "natural"}
        if self.kind == "fifo":
            return {"order_strategy": "fifo"}
        if self.kind == "dynamic":
            d = {"order_strategy": "dynamic_ready"}
            if self.yield_prior is not None:
                d["dynamic_yield_prior"] = self.yield_prior
            return d
        if self.kind == "custom":
            return {"order_strategy": "custom", "custom_order": list(self.order)}
        # policy-based orders are computed in scheduling.py (needs probes / policy)
        from . import scheduling
        order = scheduling.resolve_order(self.kind, ctx, rp, seed=self.seed)
        return {"order_strategy": "custom", "custom_order": order}


# ---------------------------------------------------------------------------
# Decision — how competing P_out values on a functional key resolve
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Decision:
    kind: str                       # canon | ec | two_phase
    scope: str = "task"             # two_phase scope: task | global
    final_canon: bool = True        # two_phase: keep the forced final canon?

    @classmethod
    def monotone(cls) -> "Decision":
        """Monotone chase, NO canonical pick — every candidate value kept (the raw
        multi-valued output; used when measuring steps/runtime, not resolution)."""
        return cls("monotone")

    @classmethod
    def canon(cls) -> "Decision":
        """Monotone chase + max-PCA canonical pick (confluent)."""
        return cls("canon")

    @classmethod
    def ec(cls) -> "Decision":
        """Early-commit, first-writer-wins (the schedule-dependent foil)."""
        return cls("ec")

    @classmethod
    def two_phase(cls, scope: str = "task", final_canon: bool = True) -> "Decision":
        return cls("two_phase", scope=scope, final_canon=final_canon)

    def build(self) -> dict:
        if self.kind == "monotone":
            return {"decision_mode": "monotone", "canonicalize": False}
        if self.kind == "canon":
            return {"decision_mode": "monotone", "canonicalize": True}
        if self.kind == "ec":
            return {"decision_mode": "early_commit", "canonicalize": False}
        if self.kind == "two_phase":
            return {"decision_mode": "two_phase", "two_phase_scope": self.scope,
                    "no_final_canon": not self.final_canon}
        raise ValueError(f"unknown decision kind {self.kind!r}")


# ---------------------------------------------------------------------------
# Method — a named comparison arm
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Method:
    """One comparison arm. `runner` selects the engine entry-point:
        'aff'     — the task-aware shared-overlay chase (AFFRunner);
        'ff'      — a flat single-task chase (sugar for partition=all_rules);
        'ff_task' — N fresh full-set re-runs, no reuse (the wasteful baseline)."""
    name: str
    partition: Partition = field(default_factory=Partition.scc_blocks)
    schedule: Schedule = field(default_factory=Schedule.natural)
    decision: Decision = field(default_factory=Decision.canon)
    runner: str = "aff"
    seed: int = 42
    max_rounds: int = 0
    record_events: bool = True
    shuffle_seed: Optional[int] = None
    slots: Optional[int] = None      # ff_task only: # of fresh re-runs (default = #tasks)
    to_quiescence: bool = False      # repeat passes until no new fact
    max_passes: int = 6              # outer pass cap when to_quiescence
