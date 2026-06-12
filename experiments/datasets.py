"""datasets — register the bundled `dbpedia_gamma` dataset (the paper's DBpedia graph +
mined rules) so experiments run against this project's own data.
"""

from __future__ import annotations

from pathlib import Path

# Top-level datasets/ dir (sibling of experiments/ and metachase/).
GAMMA = Path(__file__).resolve().parents[1] / "datasets" / "dbpedia_gamma"

_DONE = False


def register_paper_datasets() -> list:
    """Register the bundled DBpedia dataset from `datasets/dbpedia_gamma/`. Idempotent."""
    global _DONE
    from metachase.rules.dataset_config import DatasetConfig, register, list_datasets
    if _DONE:
        return list_datasets()
    proc = GAMMA / "processed"
    register(DatasetConfig(
        name="dbpedia_gamma",
        graph_dir=proc,
        rules_json=GAMMA / "rules.json",
        edge_dict_path=proc / "edge_label_dict.json",
        p_out={"director", "homeStadium", "locationCountry", "owner", "stateOfOrigin"},
        min_sup=1, min_pca=0.0, path_only=False, exclude_propagation=True,
        cycle_breaker_rules=[], gt_max_rounds=15, gt_max_derived=0,
        functional_override=None,
    ))
    _DONE = True
    return list_datasets()
