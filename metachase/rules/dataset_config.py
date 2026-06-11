"""Dataset configuration registry -- the single source of truth for per-dataset settings, so the whole experiment pipeline becomes dataset-agnostic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DatasetConfig:
    """Everything an experiment needs to know about one dataset."""
    name:        str
    graph_dir:   Path                 # processed graph directory
    rules_json:  Path                 # AMIE rules JSON

    # P_out: decision predicates whose schedule-consistency we evaluate.
    p_out:       Set[str]

    # Edge-label dict used to map rule predicate names -> edge-label ids.
    edge_dict_path: Optional[Path] = None

    # Rule filtering
    min_sup:             int  = 100
    min_pca:             float = 0.0
    path_only:           bool = False
    exclude_propagation: bool = True   # drop head-in-body rules (structural)
    cycle_breaker_rules: List[str] = field(default_factory=list)  # extra multi-atom cycle rules

    # Chase / ground-truth defaults
    gt_max_rounds:  int = 15
    gt_max_derived: int = 0            # 0 = unlimited (rely on clean rule set)

    # Explicit FUNCTIONAL p_out override (predicate-name set). When given, the
    functional_override: Optional[Set[str]] = None

    # Convenience
    def resolve_paths(self) -> "DatasetConfig":
        """Make graph_dir / rules_json / edge_dict_path absolute under ROOT."""
        if not self.graph_dir.is_absolute():
            self.graph_dir = ROOT / self.graph_dir
        if not self.rules_json.is_absolute():
            self.rules_json = ROOT / self.rules_json
        if self.edge_dict_path is None:
            self.edge_dict_path = self.graph_dir / "edge_label_dict.json"
        elif not self.edge_dict_path.is_absolute():
            self.edge_dict_path = ROOT / self.edge_dict_path
        return self


# Registry

_REGISTRY = {}


def register(cfg: DatasetConfig) -> None:
    _REGISTRY[cfg.name] = cfg.resolve_paths()


def get_config(name: str) -> DatasetConfig:
    if name in _REGISTRY:
        return _REGISTRY[name]
    raise ValueError(
        f"Unknown dataset '{name}'. Registered: {sorted(_REGISTRY)}")


def list_datasets() -> List[str]:
    return sorted(_REGISTRY)


# demo -- a tiny self-contained dataset bundled with the project

_DEMO_DIR = Path(__file__).resolve().parents[2] / "sample_data" / "demo"

register(DatasetConfig(
    name="demo",
    graph_dir=_DEMO_DIR / "processed",
    rules_json=_DEMO_DIR / "rules.json",
    edge_dict_path=_DEMO_DIR / "processed" / "edge_label_dict.json",
    p_out={"locationCountry", "owner"},
    min_sup=1, min_pca=0.0, path_only=False, exclude_propagation=False,
    cycle_breaker_rules=[],
    gt_max_rounds=10, gt_max_derived=0,
))

