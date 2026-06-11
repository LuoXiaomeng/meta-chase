"""Subgraph-isomorphism matcher for GAR patterns."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Generator, List, Optional, Set, Tuple

from ..graph.gar import PatternGraph

_GraphLike = object   # Graph or GraphView — both have the same interface


# Public interface

class Matcher:
    """Finds all injective matches of a PatternGraph in a Graph or GraphView."""

    def __init__(self, graph: _GraphLike, max_matches: int = 0) -> None:
        self.graph       = graph
        self.max_matches = max_matches

    @property
    def backend(self) -> str:
        return "python"

    def match_delta(
        self,
        pattern:    PatternGraph,
        graph:      _GraphLike,
        delta_src:  "np.ndarray",
        delta_dst:  "np.ndarray",
        delta_lbl:  "np.ndarray",
    ) -> Generator[Dict[str, int], None, None]:
        """Semi-naive matching entry point."""
        yield from self._match_python(pattern, graph)

    def match_overlay(
        self,
        pattern: PatternGraph,
        view,
    ) -> Generator[Dict[str, int], None, None]:
        """Python semi-naive: yield matches of *pattern* in *view* that use at least one OVERLAY (derived/generated) edge, deduplicated by frozen match."""
        derived = getattr(view, "_derived", None)
        if derived is None or derived.num_edges == 0 or not pattern.edges:
            return
        # Index overlay edges by label for O(1) anchor lookup.
        ov_by_label: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for (s, d, l) in derived._edge_set:
            ov_by_label[l].append((s, d))
        all_ov = [(s, d) for edges in ov_by_label.values() for (s, d) in edges]

        pnodes = pattern.nodes
        sorted_vars = sorted(pnodes.keys())
        seen: Set[Tuple[int, ...]] = set()
        for pe in pattern.edges:
            sv, dv, plabel = pe.src_var, pe.dst_var, pe.label
            anchor = all_ov if plabel is None else ov_by_label.get(plabel, [])
            if not anchor:
                continue
            s_type = pnodes[sv].label
            d_type = pnodes[dv].label
            for (u, w) in anchor:
                if sv == dv:
                    if u != w:
                        continue
                elif u == w:
                    continue          # injective: distinct vars -> distinct nodes
                if s_type is not None and view.get_type(u) != s_type:
                    continue
                if d_type is not None and view.get_type(w) != d_type:
                    continue
                seed = {sv: u, dv: w}
                # Validate every pattern edge fully inside the seed (parallel edges,
                if not _seed_edges_ok(pattern, view, seed):
                    continue
                order = _order_seeded(pattern, seed)
                for m in _backtrack(pattern, view, order, len(seed),
                                    dict(seed), set(seed.values()),
                                    self.max_matches, [0]):
                    key = tuple(m[v] for v in sorted_vars)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield m

    def match(
        self,
        pattern: PatternGraph,
        graph:   _GraphLike | None = None,
    ) -> Generator[Dict[str, int], None, None]:
        """Yield all matches (var → node_id dicts) of *pattern*."""
        g = graph if graph is not None else self.graph
        yield from self._match_python(pattern, g)

    # Python VF2 backend

    def _match_python(
        self,
        pattern: PatternGraph,
        graph:   _GraphLike,
    ) -> Generator[Dict[str, int], None, None]:
        order   = _compute_order(pattern)
        partial: Dict[str, int] = {}
        used:    Set[int]       = set()
        yield from _backtrack(
            pattern, graph, order, 0,
            partial, used, self.max_matches, [0],
        )


# Internal helpers (Python VF2 — unchanged)

def _compute_order(pattern: PatternGraph) -> List[str]:
    """Return variables in matching order (most-constrained first)."""
    if not pattern.nodes:
        return []

    adj: Dict[str, List[str]] = defaultdict(list)
    for e in pattern.edges:
        adj[e.src_var].append(e.dst_var)
        adj[e.dst_var].append(e.src_var)

    degree = {v: len(adj[v]) for v in pattern.nodes}
    seed   = max(degree, key=lambda v: degree[v])

    order:  List[str] = [seed]
    placed: Set[str]  = {seed}

    while len(order) < len(pattern.nodes):
        best, best_score = None, -1
        for v in pattern.nodes:
            if v in placed:
                continue
            score = sum(1 for u in adj[v] if u in placed)
            if score > best_score:
                best, best_score = v, score
        if best is None:
            best = next(v for v in pattern.nodes if v not in placed)
        order.append(best)
        placed.add(best)

    return order


def _order_seeded(pattern: PatternGraph, seed: Dict[str, int]) -> List[str]:
    """Variable order for backtracking that starts from already-placed `seed` vars, then adds the rest most-connected-to-placed first (same heuristic as `_compute_order`)."""
    adj: Dict[str, List[str]] = defaultdict(list)
    for e in pattern.edges:
        adj[e.src_var].append(e.dst_var)
        adj[e.dst_var].append(e.src_var)
    order:  List[str] = list(seed)
    placed: Set[str]  = set(seed)
    while len(order) < len(pattern.nodes):
        best, best_score = None, -1
        for v in pattern.nodes:
            if v in placed:
                continue
            score = sum(1 for u in adj[v] if u in placed)
            if score > best_score:
                best, best_score = v, score
        if best is None:
            best = next(v for v in pattern.nodes if v not in placed)
        order.append(best)
        placed.add(best)
    return order


def _seed_edges_ok(pattern: PatternGraph, graph, seed: Dict[str, int]) -> bool:
    """Every pattern edge whose BOTH endpoints are in `seed` must hold in graph."""
    for e in pattern.edges:
        if e.src_var in seed and e.dst_var in seed:
            if not graph.has_edge(seed[e.src_var], seed[e.dst_var], e.label):
                return False
    return True


def _get_out_edges(graph, node_id: int) -> List[Tuple[int, int]]:
    if hasattr(graph, 'get_out_edges'):
        return graph.get_out_edges(node_id)
    return graph.out_adj.get(node_id, [])


def _get_in_edges(graph, node_id: int) -> List[Tuple[int, int]]:
    if hasattr(graph, 'get_in_edges'):
        return graph.get_in_edges(node_id)
    return graph.in_adj.get(node_id, [])


def _candidates(
    var:     str,
    pattern: PatternGraph,
    graph,
    partial: Dict[str, int],
) -> List[int]:
    pnode         = pattern.nodes[var]
    required_type = pnode.label

    anchor_edges = [
        (peer, elabel, direction)
        for (peer, elabel, direction) in pattern.neighbours_of(var)
        if peer in partial
    ]

    if not anchor_edges:
        if required_type is not None:
            return graph.nodes_by_type(required_type)
        return list(graph.node_type.keys())

    candidate_set: Optional[Set[int]] = None

    for peer_var, elabel, direction in anchor_edges:
        peer_id = partial[peer_var]
        if direction == "out":
            reachable = {
                src for src, l in _get_in_edges(graph, peer_id)
                if elabel is None or l == elabel
            }
        else:
            reachable = {
                dst for dst, l in _get_out_edges(graph, peer_id)
                if elabel is None or l == elabel
            }
        candidate_set = (reachable if candidate_set is None
                         else candidate_set & reachable)

    candidates = list(candidate_set) if candidate_set is not None else []

    if required_type is not None:
        candidates = [c for c in candidates if graph.get_type(c) == required_type]

    return candidates


def _is_feasible(
    var:     str,
    node_id: int,
    pattern: PatternGraph,
    graph,
    partial: Dict[str, int],
) -> bool:
    for e in pattern.edges:
        if e.src_var == var and e.dst_var in partial:
            if not graph.has_edge(node_id, partial[e.dst_var], e.label):
                return False
        elif e.dst_var == var and e.src_var in partial:
            if not graph.has_edge(partial[e.src_var], node_id, e.label):
                return False
    return True


def _backtrack(
    pattern:     PatternGraph,
    graph,
    order:       List[str],
    depth:       int,
    partial:     Dict[str, int],
    used:        Set[int],
    max_matches: int,
    counter:     List[int],
) -> Generator[Dict[str, int], None, None]:
    if depth == len(order):
        counter[0] += 1
        yield dict(partial)
        return

    var        = order[depth]
    candidates = _candidates(var, pattern, graph, partial)

    for cand_id in candidates:
        if cand_id in used:
            continue
        if not _is_feasible(var, cand_id, pattern, graph, partial):
            continue

        partial[var] = cand_id
        used.add(cand_id)

        yield from _backtrack(
            pattern, graph, order, depth + 1,
            partial, used, max_matches, counter,
        )

        del partial[var]
        used.discard(cand_id)

        if max_matches > 0 and counter[0] >= max_matches:
            return
