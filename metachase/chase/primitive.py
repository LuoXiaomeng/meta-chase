"""metachase.chase.primitive — the chase primitive shared by every coordinator."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..graph.derived import DerivedFacts
from ..graph.gar import GAR, ConclusionType
from ..graph.graph import Graph


def materialize(
    gar:     GAR,
    match:   Dict[str, int],
    graph:   Graph,
    derived: DerivedFacts,
    functional_keys: "Optional[set]" = None,
    early_commit: bool = False,
) -> Tuple[bool, Any]:
    """Apply *gar*'s conclusion under *match* to the *derived* overlay over *graph*."""
    c = gar.conclusion

    if c.type == ConclusionType.ENRICH:
        src_var, edge_label_id, dst_var = c.target
        src_id, dst_id = match[src_var], match[dst_var]
        output = (src_id, edge_label_id, dst_id)
        if functional_keys and edge_label_id in functional_keys:
            # Base authority for FUNCTIONAL keys: if the base graph already occupies
            if graph.has_out_label(src_id, edge_label_id):
                return False, output
            # Early-commit (first-writer-wins): the FIRST derived value locks the
            if early_commit and derived.has_out_label(src_id, edge_label_id):
                return False, output
        # New only if the edge doesn't exist in the base graph either.
        already = graph.has_edge(src_id, dst_id, edge_label_id)
        is_new = (not already) and derived.add_edge(src_id, dst_id, edge_label_id)
        return is_new, output

    elif c.type == ConclusionType.DEDUCE:
        var, attr, value = c.target
        node_id = match[var]
        output = (var, attr, value)
        # New only if the attribute isn't already in base or overlay.
        already = graph.get_attr(node_id, attr) is not None
        is_new = (not already) and derived.add_attr(node_id, attr, str(value))
        return is_new, output

    elif c.type == ConclusionType.CLEAN:
        # CLEAN is always "new" the first time a node is flagged.
        node_ids = tuple(match[v] for v in c.target)
        output = node_ids
        flag_attr = f"__clean__{gar.id}"
        is_new = derived.add_attr(node_ids[0], flag_attr, "1")
        return is_new, output

    else:
        raise ValueError(f"Unknown conclusion type: {c.type}")
