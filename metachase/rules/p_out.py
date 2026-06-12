"""Resolve P_out predicate names to graph edge-label IDs."""

from __future__ import annotations

from typing import Set


def resolve_p_out_ids(p_out_names: Set[str], edge_label_dict: dict) -> Set[int]:
    """Convert predicate names to edge label IDs, dropping any not in the graph."""
    ids = set()
    missing = []
    for name in p_out_names:
        if name in edge_label_dict:
            ids.add(edge_label_dict[name])
        else:
            missing.append(name)
    if missing:
        print(f"[p_out] WARNING: predicates not found in graph: {missing}")
    return ids
