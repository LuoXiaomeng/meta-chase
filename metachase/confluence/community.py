"""Confluence — V.A output-commutativity check (paper Section V.A)."""

from __future__ import annotations

from typing import List, Set, Tuple

from ..chase.coordinator import VanillaCoordinator
from ..graph.derived import DerivedFacts
from .two_phase import _collect_p_out_from_derived


# V.A  Output-commutativity check (sufficient condition)

def check_output_commutativity(tasks_meta: List[List[dict]],
                               p_out_preds: Set[str]
                               ) -> Tuple[bool, List[Tuple[int, int, str]]]:
    """Check whether a task partition is output-commutative (sufficient cond.)."""
    # Per-task head and body predicate sets
    heads = []
    bodies = []
    p_out_heads = []
    for task in tasks_meta:
        h = {r["head_atom"][1] for r in task}
        b = {a[1] for r in task for a in r["body_atoms"]}
        heads.append(h)
        bodies.append(b)
        p_out_heads.append(h & p_out_preds)

    interferences = []
    n = len(tasks_meta)
    for i in range(n):
        for j in range(i + 1, n):
            reasons = []
            # producer -> consumer either direction
            if heads[i] & bodies[j]:
                reasons.append(f"task{i} produces {sorted(heads[i] & bodies[j])} consumed by task{j}")
            if heads[j] & bodies[i]:
                reasons.append(f"task{j} produces {sorted(heads[j] & bodies[i])} consumed by task{i}")
            # competition on a P_out predicate
            shared_pout = p_out_heads[i] & p_out_heads[j]
            if shared_pout:
                reasons.append(f"both produce P_out {sorted(shared_pout)} (potential value conflict)")
            if reasons:
                interferences.append((i, j, "; ".join(reasons)))

    return (len(interferences) == 0), interferences


# V.A (exact) -- instance-level output-commutativity check (OFFLINE analysis)

def _chase_compose(graph, first_gars, second_gars, p_out_ids, matcher,
                   max_rounds, max_derived):
    """Compute Chase(second, Chase(first, G)) and return its P_out triple set."""
    derived = DerivedFacts(); fired = set()
    for gars in (first_gars, second_gars):
        c = VanillaCoordinator(graph, gars, max_rounds=max_rounds, max_matches=0,
                               max_derived=max_derived, verbose=False,
                               matcher=matcher, p_out_ids=p_out_ids)
        c.run(existing_derived=derived, existing_fired_pairs=fired, task_id="CC")
    return _collect_p_out_from_derived(derived, p_out_ids)


def check_community_exact(graph, tasks_gars, tasks_meta, p_out_ids, p_out_preds,
                          matcher, max_rounds=10, max_derived=0):
    """Exact output-commutativity verdict on graph G (OFFLINE analysis)."""
    is_struct_comm, interf = check_output_commutativity(tasks_meta, p_out_preds)
    if is_struct_comm:
        # Structurally independent -> output-commutative (Theorem 3). No chase.
        return {'structural_commutative': True, 'exact_commutative': True,
                'checked_pairs': [], 'divergent_pairs': []}

    # Only the structurally-interfering pairs need the (expensive) chase test.
    candidate_pairs = [(i, j) for (i, j, _) in interf]
    divergent = []
    for (i, j) in candidate_pairs:
        out_ij = _chase_compose(graph, tasks_gars[i], tasks_gars[j],
                                p_out_ids, matcher, max_rounds, max_derived)
        out_ji = _chase_compose(graph, tasks_gars[j], tasks_gars[i],
                                p_out_ids, matcher, max_rounds, max_derived)
        if out_ij != out_ji:
            divergent.append((i, j, len(out_ij - out_ji), len(out_ji - out_ij)))

    return {'structural_commutative': False,
            'exact_commutative': (len(divergent) == 0),
            'checked_pairs': candidate_pairs,
            'divergent_pairs': divergent}
