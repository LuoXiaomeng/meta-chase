"""DerivedFacts — overlay of edges and attributes produced by GAR firings."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .graph import Graph


# DerivedFacts

class DerivedFacts:
    """Stores edges and node-attribute values derived by GAR conclusions."""

    def __init__(self) -> None:
        # Adjacency for derived edges
        self.out_adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.in_adj:  Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        # Derived attributes: node_id → {attr: value}
        self.attrs: Dict[int, Dict[str, str]] = defaultdict(dict)

        # Deduplication sets
        self._edge_set: Set[Tuple[int, int, int]] = set()
        self._attr_set: Set[Tuple[int, str]]      = set()

        # Edges / attribute names added since last call to take_delta_*.
        self._round_edges: List[Tuple[int, int, int]] = []
        self._round_attrs: Set[str] = set()   # attr names set this round

        # Monotonic version counter: incremented on every successful
        self._version: int = 0

        # Growable numpy arrays mirroring _edge_set (edges are append-only, never
        self._oa_src = None        # uint32, capacity self._oa_cap
        self._oa_dst = None
        self._oa_lbl = None        # int32
        self._oa_cap = 0           # allocated capacity (>= num_edges)
        # Count of remove_edge calls. Stays 0 for monotone chases (canon / two-phase /
        self._removed_count = 0

    def clone(self) -> "DerivedFacts":
        """Deep-enough copy for non-committing probes (Γ-aware greedy policy): mutating the clone (add_edge/add_attr) never touches this object."""
        new = DerivedFacts()
        new._edge_set = set(self._edge_set)
        new.out_adj = defaultdict(list, {k: list(v) for k, v in self.out_adj.items()})
        new.in_adj = defaultdict(list, {k: list(v) for k, v in self.in_adj.items()})
        new.attrs = defaultdict(dict, {k: dict(v) for k, v in self.attrs.items()})
        new._attr_set = set(self._attr_set)
        new._round_edges = list(self._round_edges)
        new._round_attrs = set(self._round_attrs)
        new._version = self._version
        new._oa_cap = self._oa_cap
        new._removed_count = self._removed_count   # keep _oa_* staleness flag in sync
        if self._oa_src is not None:
            new._oa_src = self._oa_src.copy()
            new._oa_dst = self._oa_dst.copy()
            new._oa_lbl = self._oa_lbl.copy()
        return new

    # Writers (return True if the fact was genuinely new)

    def add_edge(self, src: int, dst: int, label: int) -> bool:
        """Add a derived directed edge."""
        key = (src, dst, label)
        if key in self._edge_set:
            return False
        self._edge_set.add(key)
        self.out_adj[src].append((dst, label))
        self.in_adj[dst].append((src, label))
        self._round_edges.append(key)
        self._oa_append(src, dst, label)
        self._version += 1
        return True

    def _oa_append(self, src: int, dst: int, label: int) -> None:
        """Append one edge to the growable overlay arrays (O(1) amortized)."""
        import numpy as np
        n = len(self._edge_set) - 1            # index for this edge (we already added to set)
        if n >= self._oa_cap:
            new_cap = 1024 if self._oa_cap == 0 else self._oa_cap * 2
            ns = np.empty(new_cap, dtype=np.uint32)
            nd = np.empty(new_cap, dtype=np.uint32)
            nl = np.empty(new_cap, dtype=np.int32)
            if self._oa_cap:
                ns[:n] = self._oa_src[:n]
                nd[:n] = self._oa_dst[:n]
                nl[:n] = self._oa_lbl[:n]
            self._oa_src, self._oa_dst, self._oa_lbl = ns, nd, nl
            self._oa_cap = new_cap
        self._oa_src[n] = src
        self._oa_dst[n] = dst
        self._oa_lbl[n] = label

    def remove_edge(self, src: int, dst: int, label: int) -> bool:
        """Remove a derived edge in place."""
        key = (src, dst, label)
        if key not in self._edge_set:
            return False
        self._edge_set.discard(key)
        oa = self.out_adj.get(src)
        if oa:
            try:
                oa.remove((dst, label))
            except ValueError:
                pass
        ia = self.in_adj.get(dst)
        if ia:
            try:
                ia.remove((src, label))
            except ValueError:
                pass
        self._version += 1
        self._removed_count += 1   # marks _oa_* stale -> serialize_arrays rebuilds
        return True

    def serialize_arrays(self):
        """Return (src, dst, lbl) numpy arrays of the overlay edges, length = num_edges."""
        import numpy as np
        n = len(self._edge_set)
        if n == 0:
            return (np.empty(0, np.uint32), np.empty(0, np.uint32), np.empty(0, np.int32))
        if self._removed_count == 0:
            # fast monotone path: _oa_* mirrors _edge_set in insertion order (O(1) slice).
            return (self._oa_src[:n], self._oa_dst[:n], self._oa_lbl[:n])
        # remove-aware path — ONLY taken after a mid-chase collapse (remove_edge). The
        src = np.empty(n, np.uint32)
        dst = np.empty(n, np.uint32)
        lbl = np.empty(n, np.int32)
        for i, (s, d, l) in enumerate(self._edge_set):
            src[i] = s; dst[i] = d; lbl[i] = l
        return (src, dst, lbl)

    def edges_since(self, watermark: int) -> List[Tuple[int, int, int]]:
        """Return overlay edges added AFTER index `watermark` (insertion order), as a list of (src, dst, label)."""
        n = len(self._edge_set)
        w = max(0, min(watermark, n))
        if w >= n:
            return []
        return [(int(self._oa_src[i]), int(self._oa_dst[i]), int(self._oa_lbl[i]))
                for i in range(w, n)]

    def take_delta_edges(self) -> List[Tuple[int, int, int]]:
        """Return edges added since the last call and reset the buffer."""
        delta = self._round_edges
        self._round_edges = []
        return delta

    def take_delta_attrs(self) -> "Set[str]":
        """Return attribute names set since the last call and reset the buffer."""
        delta = self._round_attrs
        self._round_attrs = set()
        return delta

    def add_attr(self, node_id: int, attr: str, value: str) -> bool:
        """Add a derived attribute value."""
        key = (node_id, attr)
        if key in self._attr_set:
            return False
        self._attr_set.add(key)
        self.attrs[node_id][attr] = value
        self._round_attrs.add(attr)
        self._version += 1
        return True

    # Readers

    def has_edge(self, src: int, dst: int,
                 label: Optional[int] = None) -> bool:
        if label is None:
            return any(d == dst for d, _ in self.out_adj.get(src, []))
        return (src, dst, label) in self._edge_set

    def has_out_label(self, src: int, label: int) -> bool:
        """True if the overlay already has ANY out-edge (src, *, label)."""
        return any(l == label for _, l in self.out_adj.get(src, []))

    def get_attr(self, node_id: int, attr: str) -> Optional[str]:
        return self.attrs.get(node_id, {}).get(attr)

    @property
    def num_edges(self) -> int:
        return len(self._edge_set)

    @property
    def num_attrs(self) -> int:
        return len(self._attr_set)


# GraphView

class GraphView:
    """Read-only view of a base Graph combined with a DerivedFacts overlay."""

    def __init__(self, base: Graph, derived: DerivedFacts) -> None:
        self._base    = base
        self._derived = derived

        # Expose dicts that Graph code accesses directly
        self.node_type       = base.node_type
        self.node_type_dict  = base.node_type_dict
        self.edge_label_dict = base.edge_label_dict
        self.inv_node_type   = base.inv_node_type
        self.inv_edge_label  = base.inv_edge_label
        self.num_vertices    = base.num_vertices

    # Adjacency — combined original + derived

    def get_out_edges(self, node_id: int) -> List[Tuple[int, int]]:
        """All outgoing (dst, label) pairs for *node_id*."""
        base    = self._base.out_adj.get(node_id, [])
        derived = self._derived.out_adj.get(node_id, [])
        return base + derived if derived else base

    def get_in_edges(self, node_id: int) -> List[Tuple[int, int]]:
        """All incoming (src, label) pairs for *node_id*."""
        base    = self._base.in_adj.get(node_id, [])
        derived = self._derived.in_adj.get(node_id, [])
        return base + derived if derived else base

    def has_edge(self, src: int, dst: int,
                 label: Optional[int] = None) -> bool:
        return (self._base.has_edge(src, dst, label) or
                self._derived.has_edge(src, dst, label))

    def has_in_edge(self, dst: int, src: int,
                    label: Optional[int] = None) -> bool:
        return self.has_edge(src, dst, label)

    # Node queries

    def get_type(self, node_id: int) -> Optional[int]:
        return self._base.get_type(node_id)

    def type_name(self, node_id: int) -> str:
        return self._base.type_name(node_id)

    def edge_name(self, label_id: int) -> str:
        return self._base.edge_name(label_id)

    def nodes_by_type(self, type_id: int) -> List[int]:
        return self._base.nodes_by_type(type_id)

    def out_degree(self, node_id: int) -> int:
        return (self._base.out_degree(node_id) +
                len(self._derived.out_adj.get(node_id, [])))

    def in_degree(self, node_id: int) -> int:
        return (self._base.in_degree(node_id) +
                len(self._derived.in_adj.get(node_id, [])))

    def get_attr(self, node_id: int, attr: str,
                 default: Optional[str] = None) -> Optional[str]:
        """Check derived attrs first (more recent), then base graph."""
        v = self._derived.get_attr(node_id, attr)
        if v is not None:
            return v
        return self._base.get_attr(node_id, attr, default)
