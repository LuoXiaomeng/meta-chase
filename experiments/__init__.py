"""experiments — a thin declarative harness over the `metachase` engine. An experiment
declares which engine capabilities to compare (a Method = partition × schedule ×
decision × runner) and how to plot them; removing `experiments/` leaves a working engine.
"""

from .spec import Method, Partition, Schedule, Decision

__all__ = ["Method", "Partition", "Schedule", "Decision"]
