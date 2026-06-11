"""AFF (Agentized Fishing Fort) — single-entry facade for the policy-driven task-aware GAR-subset chase."""

from .config import AFFConfig
from .runner import AFFRunner, AFFResult
from .baselines import run_ff, run_ff_task

__all__ = ["AFFConfig", "AFFRunner", "AFFResult", "run_ff", "run_ff_task"]
