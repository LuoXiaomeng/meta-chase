"""metrics — named `(trace, gt) -> number` extractors selected per experiment. Chasing steps
and rule invocations are backend-invariant; runtime is backend-sensitive; recall needs the GT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Set

from .pipeline import Trace


@dataclass(frozen=True)
class Metric:
    key: str                                    # JSON key + plot lookup
    label: str                                  # axis label
    extract: Callable[[Trace, Optional[Set]], float]
    needs_gt: bool = False

    def __call__(self, trace: Trace, gt: Optional[Set] = None) -> float:
        return self.extract(trace, gt)


# -- backend-invariant cost metrics ---------------------------------------
Metric.steps = Metric("steps", "chasing steps", lambda t, gt: t.total_steps)
Metric.invocations = Metric("inv", "rule invocations", lambda t, gt: t.invocations)
Metric.runtime = Metric("time", "runtime (s)", lambda t, gt: round(t.runtime, 4))
Metric.task_count = Metric("tasks", "#tasks", lambda t, gt: t.n_tasks)
Metric.p_out_size = Metric("p_out", "|P_out|", lambda t, gt: len(t.p_out))


# -- GT-dependent quality metrics -----------------------------------------
def _recall(trace: Trace, gt: Optional[Set]) -> float:
    if not gt:
        return float("nan")
    return len(trace.p_out & gt) / len(gt)


def _precision(trace: Trace, gt: Optional[Set]) -> float:
    if not trace.p_out or not gt:
        return float("nan")
    return len(trace.p_out & gt) / len(trace.p_out)


Metric.recall = Metric("recall", "recall", _recall, needs_gt=True)
Metric.precision = Metric("precision", "precision", _precision, needs_gt=True)
