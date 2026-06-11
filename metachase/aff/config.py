"""AFFConfig — the single configuration object enumerating every AFF knob."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class AFFConfig:
    """All configurable knobs of the AFF system, in one place."""

    # 1. Task source — what is the unit task?
    task_source: str = "scc_blocks"
    prebuilt_tasks: Optional[List] = None     # used only when task_source='prebuilt'

    # 2. Task order — how are tasks sequenced?
    order_strategy: str = "natural"
    custom_order: Optional[List[int]] = None
    seed: int = 42

    # 3. Cross-task sharing — how is the IRS (Intermediate Result Set) reused?
    sharing: str = "shared"

    # 4. Termination
    to_quiescence: bool = False             # repeat passes until no new fact
    # When a block is RE-invoked in the to_quiescence outer loop, seed it with the
    reinvoke_incremental: bool = False
    max_passes: int = 6                     # outer pass cap when to_quiescence=True
    max_rounds: int = 0                     # per-coord round cap (0 = run to fixpoint, no cap)
    max_matches: int = 0                    # per-rule-per-round match cap (0 = unlimited)
    max_derived: int = 0                    # global derived-fact cap (0 = unlimited)

    # 5. Decision mode — how are FUNCTIONAL P_out conflicts resolved?
    decision_mode: str = "monotone"
    # Scope of the two-phase protocol (only consulted when decision_mode='two_phase'):
    two_phase_scope: str = "global"

    # 6. Post-processing
    canonicalize: bool = False              # apply max-PCA decision picker over all producers
    # Explicit opt-out of the FINAL canonicalize, even under decision_mode='two_phase'
    no_final_canon: bool = False
    base_authority: bool = True             # functional P_out key in base => don't derive
    # Optional per-task yield prior (indexed by task) used by order_strategy=
    dynamic_yield_prior: Optional[List[float]] = None
    # When set, shuffle each task's rule (GAR) firing order with this seed before the
    shuffle_rules_seed: Optional[int] = None
    # Optional per-task historical statistics {task_id: (gain, cost)} used to WARM
    warmup_stats: Optional[dict] = None

    # 7. Diagnostics
    verbose: bool = False
    task_id_fmt: str = "C{i}"               # how AFFRunner tags per-task FiringResults
    # When True, the coordinator records an ordered, timestamped P_out event log
    record_p_out_events: bool = False
    # When True, populate AFFResult.per_task_snapshots with the cumulative
    track_snapshots: bool = False

    # Currently-implemented value sets. Each new phase appends to these.
    SUPPORTED_TASK_SOURCES = frozenset({
        # Phase 1
        "scc_blocks",
        # Phase 2
        "all_rules", "all_rules_focused",
        # Phase 6: caller-supplied task list (block_type grid etc.)
        "prebuilt",
    })
    SUPPORTED_ORDER_STRATEGIES = frozenset({
        # Phase 1
        "natural", "custom",
        # FIFO: fixed topological order, earliest-eligible task first.
        "fifo",
        # Γ-aware DYNAMIC readiness: at each step pick the task whose consumed
        "dynamic_ready",
    })
    SUPPORTED_SHARING = frozenset({
        # Phase 1
        "shared",
        # Phase 2
        "fresh",
        # 'task_isolated' arrives later if useful
    })
    SUPPORTED_DECISION_MODES = frozenset({
        # Phase 1
        "monotone",
        # Phase 3
        "early_commit", "two_phase",
    })

    def validate(self) -> None:
        """Raise NotImplementedError if config touches a knob value not yet wired."""
        errs = []
        if self.task_source not in self.SUPPORTED_TASK_SOURCES:
            errs.append(f"task_source={self.task_source!r} unsupported (have: {sorted(self.SUPPORTED_TASK_SOURCES)})")
        if self.order_strategy not in self.SUPPORTED_ORDER_STRATEGIES:
            errs.append(f"order_strategy={self.order_strategy!r} unsupported (have: {sorted(self.SUPPORTED_ORDER_STRATEGIES)})")
        if self.sharing not in self.SUPPORTED_SHARING:
            errs.append(f"sharing={self.sharing!r} unsupported (have: {sorted(self.SUPPORTED_SHARING)})")
        if self.decision_mode not in self.SUPPORTED_DECISION_MODES:
            errs.append(f"decision_mode={self.decision_mode!r} unsupported (have: {sorted(self.SUPPORTED_DECISION_MODES)})")
        # Coherence checks
        if self.order_strategy == "custom" and self.custom_order is None:
            errs.append("order_strategy='custom' requires custom_order")
        if self.two_phase_scope not in ("global", "task"):
            errs.append(f"two_phase_scope={self.two_phase_scope!r} must be 'global' or 'task'")
        if errs:
            raise NotImplementedError(
                "AFFConfig touches unimplemented or invalid knobs:\n  " + "\n  ".join(errs)
            )
