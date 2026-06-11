"""DeltaMatcher — incremental subgraph-isomorphism matching."""

from __future__ import annotations

from typing import Dict, FrozenSet, Generator, List, Optional, Set, Tuple

from ..graph.gar import PatternGraph
from .matcher import (
    _backtrack, _candidates, _compute_order, _is_feasible,
    _get_out_edges, _get_in_edges,
)

_GraphLike = object   # Graph or GraphView


# Public interface

def match_delta(
    pattern:      PatternGraph,
    view:         _GraphLike,
    delta_edges:  Set[Tuple[int, int, int]],   # (src, dst, label) triples
    max_matches:  int = 0,
) -> Generator[Dict[str, int], None, None]:
    """Yield all matches of *pattern* in *view* that use ≥1 edge from *delta_edges*."""
    if not delta_edges or not pattern.nodes:
        return

    order   = _compute_order(pattern)
    seen:   Set[FrozenSet] = set()
    counter = [0]

    for pe in pattern.edges:
        # pe.label == None means wildcard — match any label
        for (s, d, lbl) in delta_edges:
            if pe.label is not None and lbl != pe.label:
                continue

            # Check type constraints for both anchor nodes
            src_type = pattern.nodes[pe.src_var].label
            dst_type = pattern.nodes[pe.dst_var].label
            if src_type is not None and view.get_type(s) != src_type:
                continue
            if dst_type is not None and view.get_type(d) != dst_type:
                continue

            # Injective: anchor nodes must be distinct
            if s == d:
                continue

            # Build the partial assignment for the anchor edge
            partial: Dict[str, int] = {pe.src_var: s, pe.dst_var: d}
            used:    Set[int]       = {s, d}

            # Verify the anchor edge is feasible (both directions are consistent
            if not _partial_feasible(pe.src_var, s, pattern, view, {pe.dst_var: d}):
                continue
            if not _partial_feasible(pe.dst_var, d, pattern, view, partial):
                continue

            # Build a new order that puts already-fixed vars first
            fixed    = [v for v in order if v in partial]
            unfixed  = [v for v in order if v not in partial]
            new_order = fixed + unfixed
            depth    = len(fixed)   # start backtracking from first unfixed slot

            for m in _backtrack(
                pattern, view, new_order, depth,
                partial, used, max_matches, counter,
            ):
                key = frozenset(m.items())
                if key not in seen:
                    seen.add(key)
                    yield m

            if max_matches > 0 and counter[0] >= max_matches:
                return


# Internal helpers

def _partial_feasible(
    var:     str,
    node_id: int,
    pattern: PatternGraph,
    graph:   _GraphLike,
    partial: Dict[str, int],
) -> bool:
    """Check all pattern edges between *var* and already-placed variables."""
    for e in pattern.edges:
        if e.src_var == var and e.dst_var in partial:
            dst_id = partial[e.dst_var]
            if not graph.has_edge(node_id, dst_id, e.label):
                return False
        elif e.dst_var == var and e.src_var in partial:
            src_id = partial[e.src_var]
            if not graph.has_edge(src_id, node_id, e.label):
                return False
    return True
