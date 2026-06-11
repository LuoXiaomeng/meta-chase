"""metachase.rules — input-prep layer that builds the engine's rule/task views."""

from .amie_loader import load_amie_gars
from .group_rules import compute_scc_blocks, relevant_rules
from .dataset_config import DatasetConfig, get_config, list_datasets, register
from .p_out import resolve_p_out_ids

__all__ = [
    "load_amie_gars",
    "compute_scc_blocks", "relevant_rules",
    "DatasetConfig", "get_config", "list_datasets", "register",
    "resolve_p_out_ids",
]
