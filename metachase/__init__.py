"""metachase — Agentic graph-association-rule (GAR) Meta-Chase engine."""

from .graph import Graph, GAR, load_processed
from .match import Matcher
from .aff import AFFRunner, AFFConfig, AFFResult, run_ff, run_ff_task
from .partition import build_merged_scc_tasks, build_random_partition_tasks
from .session import Context

__all__ = [
    "Context",
    "AFFRunner", "AFFConfig", "AFFResult",
    "run_ff", "run_ff_task",
    "build_merged_scc_tasks", "build_random_partition_tasks",
    "Graph", "GAR", "load_processed", "Matcher",
]
