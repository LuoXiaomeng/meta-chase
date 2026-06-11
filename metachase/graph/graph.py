"""In-memory graph representation for pure-Python GAR matching."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


# CSR-backed proxy that emulates the dict-of-list adjacency interface used

class _CSRAdjacencyView:
    """Read-only dict-like view over a CSR-encoded adjacency."""

    __slots__ = ("_offsets", "_neighbors", "_labels", "_n", "_nonempty_count")

    def __init__(self, offsets, neighbors, labels) -> None:
        # offsets[v]   : start index in neighbors/labels for vertex v
        self._offsets   = offsets
        self._neighbors = neighbors
        self._labels    = labels
        self._n         = len(offsets) - 1
        # Lazy: counted on demand because __len__ is rarely on hot path.
        self._nonempty_count: Optional[int] = None

    def __getitem__(self, v: int) -> List[Tuple[int, int]]:
        return self.get(v, [])

    def get(self, v: int, default: Any = None) -> List[Tuple[int, int]]:
        if v < 0 or v >= self._n:
            return [] if default is None else default
        s = int(self._offsets[v])
        e = int(self._offsets[v + 1])
        if s == e:
            return [] if default is None else default
        # tolist() converts numpy slices to Python lists in C; zip+list is fast.
        return list(zip(self._neighbors[s:e].tolist(),
                        self._labels[s:e].tolist()))

    def __contains__(self, v: int) -> bool:
        if v < 0 or v >= self._n:
            return False
        return int(self._offsets[v + 1]) > int(self._offsets[v])

    def __len__(self) -> int:
        if self._nonempty_count is None:
            import numpy as np
            self._nonempty_count = int(
                (np.diff(self._offsets) > 0).sum()
            )
        return self._nonempty_count

    def items(self) -> Iterator[Tuple[int, List[Tuple[int, int]]]]:
        offs = self._offsets
        nbrs = self._neighbors
        lbls = self._labels
        for v in range(self._n):
            s = int(offs[v])
            e = int(offs[v + 1])
            if s == e:
                continue
            yield v, list(zip(nbrs[s:e].tolist(), lbls[s:e].tolist()))

    def values(self) -> Iterator[List[Tuple[int, int]]]:
        for _, vs in self.items():
            yield vs

    def keys(self) -> Iterator[int]:
        offs = self._offsets
        for v in range(self._n):
            if int(offs[v + 1]) > int(offs[v]):
                yield v


class _ArrayBackedTypeDict:
    """Dict-like view of node_id → type_id backed by a single numpy array."""

    __slots__ = ("_arr", "_n", "_count")

    def __init__(self, arr) -> None:
        self._arr   = arr           # int32 array of length num_vertices
        self._n     = len(arr)
        self._count: Optional[int] = None

    def get(self, v: int, default: Any = None) -> Any:
        if v < 0 or v >= self._n:
            return default
        val = int(self._arr[v])
        return val if val >= 0 else default

    def __getitem__(self, v: int) -> int:
        if v < 0 or v >= self._n:
            raise KeyError(v)
        val = int(self._arr[v])
        if val < 0:
            raise KeyError(v)
        return val

    def __contains__(self, v: int) -> bool:
        if v < 0 or v >= self._n:
            return False
        return int(self._arr[v]) >= 0

    def __len__(self) -> int:
        if self._count is None:
            import numpy as np
            self._count = int((self._arr >= 0).sum())
        return self._count

    def items(self) -> Iterator[Tuple[int, int]]:
        arr = self._arr
        for v in range(self._n):
            t = int(arr[v])
            if t >= 0:
                yield v, t

    def values(self) -> Iterator[int]:
        arr = self._arr
        for v in range(self._n):
            t = int(arr[v])
            if t >= 0:
                yield t

    def keys(self) -> Iterator[int]:
        arr = self._arr
        for v in range(self._n):
            if int(arr[v]) >= 0:
                yield v


class Graph:
    """Directed property graph loaded from pre-processed CSV files."""

    def __init__(self) -> None:
        self.num_vertices: int = 0
        self.num_edges:    int = 0

        # node_id → type_id  (0 = unknown)
        self.node_type:  Dict[int, int] = {}

        # node_id → {attr_name: value}  (empty dict if no attrs file)
        self.node_attrs: Dict[int, Dict[str, str]] = defaultdict(dict)

        # adjacency lists
        self.out_adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.in_adj:  Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        # label dictionaries
        self.node_type_dict:  Dict[str, int] = {}
        self.edge_label_dict: Dict[str, int] = {}
        self.inv_node_type:   Dict[int, str] = {}
        self.inv_edge_label:  Dict[int, str] = {}

    # Factory

    # CSR compaction

    def _compact_to_csr(self) -> None:
        """Convert out_adj / in_adj / node_type from dicts to CSR + proxies."""
        import numpy as np

        n_v = self.num_vertices
        if n_v == 0:
            # Empty graph -- still create well-formed empty CSR.
            self._out_offsets = np.zeros(1, dtype=np.int64)
            self._out_dsts    = np.empty(0, dtype=np.uint32)
            self._out_lbls    = np.empty(0, dtype=np.int32)
            self._in_offsets  = np.zeros(1, dtype=np.int64)
            self._in_srcs     = np.empty(0, dtype=np.uint32)
            self._in_lbls     = np.empty(0, dtype=np.int32)
            self._node_type_array = np.full(0, -1, dtype=np.int32)
            self.out_adj   = _CSRAdjacencyView(self._out_offsets,
                                               self._out_dsts, self._out_lbls)
            self.in_adj    = _CSRAdjacencyView(self._in_offsets,
                                               self._in_srcs, self._in_lbls)
            self.node_type = _ArrayBackedTypeDict(self._node_type_array)
            return

        # --- out adjacency --------------------------------------------------
        out_deg = np.zeros(n_v, dtype=np.int64)
        for src, edges in self.out_adj.items():
            if 0 <= src < n_v:
                out_deg[src] = len(edges)
        self._out_offsets = np.empty(n_v + 1, dtype=np.int64)
        self._out_offsets[0] = 0
        np.cumsum(out_deg, out=self._out_offsets[1:])

        n_e_out = int(self._out_offsets[-1])
        self._out_dsts = np.empty(n_e_out, dtype=np.uint32)
        self._out_lbls = np.empty(n_e_out, dtype=np.int32)
        # Fill by walking the original dict; preserves per-vertex order.
        cursor = self._out_offsets[:-1].copy()
        for src, edges in self.out_adj.items():
            if not (0 <= src < n_v):
                continue
            idx = int(cursor[src])
            for dst, label in edges:
                self._out_dsts[idx] = dst
                self._out_lbls[idx] = label
                idx += 1

        # --- in adjacency ---------------------------------------------------
        in_deg = np.zeros(n_v, dtype=np.int64)
        for dst, edges in self.in_adj.items():
            if 0 <= dst < n_v:
                in_deg[dst] = len(edges)
        self._in_offsets = np.empty(n_v + 1, dtype=np.int64)
        self._in_offsets[0] = 0
        np.cumsum(in_deg, out=self._in_offsets[1:])

        n_e_in = int(self._in_offsets[-1])
        self._in_srcs = np.empty(n_e_in, dtype=np.uint32)
        self._in_lbls = np.empty(n_e_in, dtype=np.int32)
        cursor = self._in_offsets[:-1].copy()
        for dst, edges in self.in_adj.items():
            if not (0 <= dst < n_v):
                continue
            idx = int(cursor[dst])
            for src, label in edges:
                self._in_srcs[idx] = src
                self._in_lbls[idx] = label
                idx += 1

        # --- node_type to numpy array ---------------------------------------
        self._node_type_array = np.full(n_v, -1, dtype=np.int32)
        for nid, tid in self.node_type.items():
            if 0 <= nid < n_v:
                self._node_type_array[nid] = tid

        # --- swap public attributes to proxies ------------------------------
        self.out_adj   = _CSRAdjacencyView(self._out_offsets,
                                           self._out_dsts, self._out_lbls)
        self.in_adj    = _CSRAdjacencyView(self._in_offsets,
                                           self._in_srcs, self._in_lbls)
        self.node_type = _ArrayBackedTypeDict(self._node_type_array)

        # Pre-populate the matcher's base-arrays cache so the first match
        v_id = np.arange(n_v, dtype=np.uint32)
        # Convert offsets diff into per-edge src
        diffs = np.diff(self._out_offsets).astype(np.int64)
        e_src = np.repeat(np.arange(n_v, dtype=np.uint32), diffs)
        # Flat per-edge arrays (e_src[i]/e_dst[i]/e_label[i] describe edge i),
        self._mg_base_arrays_cache = (
            v_id,
            self._node_type_array,
            e_src,
            self._out_dsts,
            self._out_lbls,
        )

    # Binary cache (npz) helpers

    def _save_npz_cache(self, path: Path) -> None:
        """Serialize the loaded Graph (post-CSR-compaction) to a .npz file."""
        import numpy as np

        # Node attrs: flatten to three parallel arrays.
        attr_nids:   List[int] = []
        attr_names:  List[str] = []
        attr_values: List[str] = []
        for nid, kv in self.node_attrs.items():
            for k, v in kv.items():
                attr_nids.append(nid)
                attr_names.append(k)
                attr_values.append(v)
        attr_nids_a   = np.array(attr_nids,   dtype=np.int64)
        attr_names_a  = np.array(attr_names,  dtype=object)
        attr_values_a = np.array(attr_values, dtype=object)

        np.savez(
            path,
            # CSR storage -- the new source of truth.
            out_offsets=self._out_offsets,
            out_dsts=self._out_dsts,
            out_lbls=self._out_lbls,
            in_offsets=self._in_offsets,
            in_srcs=self._in_srcs,
            in_lbls=self._in_lbls,
            node_type_array=self._node_type_array,
            # Side data.
            attr_nids=attr_nids_a,
            attr_names=attr_names_a,
            attr_values=attr_values_a,
            node_type_dict=json.dumps(self.node_type_dict),
            edge_label_dict=json.dumps(self.edge_label_dict),
            num_vertices=np.int64(self.num_vertices),
            num_edges=np.int64(self.num_edges),
        )

    @classmethod
    def _from_npz_cache(cls, path: Path, csv_dir: Path) -> "Graph":
        """Rebuild a Graph from a v2 .npz cache (CSR arrays already laid out)."""
        import numpy as np
        z = np.load(path, allow_pickle=True)

        g = cls()
        g.node_type_dict  = json.loads(str(z['node_type_dict']))
        g.edge_label_dict = json.loads(str(z['edge_label_dict']))
        g.inv_node_type   = {v: k for k, v in g.node_type_dict.items()}
        g.inv_edge_label  = {v: k for k, v in g.edge_label_dict.items()}

        g.num_vertices = int(z['num_vertices'])
        g.num_edges    = int(z['num_edges'])

        # CSR arrays -- load and immediately wire up proxies (skip dict build).
        g._out_offsets    = z['out_offsets']
        g._out_dsts       = z['out_dsts']
        g._out_lbls       = z['out_lbls']
        g._in_offsets     = z['in_offsets']
        g._in_srcs        = z['in_srcs']
        g._in_lbls        = z['in_lbls']
        g._node_type_array = z['node_type_array']

        g.out_adj   = _CSRAdjacencyView(g._out_offsets, g._out_dsts, g._out_lbls)
        g.in_adj    = _CSRAdjacencyView(g._in_offsets,  g._in_srcs,  g._in_lbls)
        g.node_type = _ArrayBackedTypeDict(g._node_type_array)

        # Pre-populate the matcher base-arrays cache for free first match.
        n_v = g.num_vertices
        v_id = np.arange(n_v, dtype=np.uint32)
        diffs = np.diff(g._out_offsets).astype(np.int64)
        e_src = np.repeat(np.arange(n_v, dtype=np.uint32), diffs)
        g._mg_base_arrays_cache = (
            v_id,
            g._node_type_array,
            e_src,
            g._out_dsts,
            g._out_lbls,
        )

        # Node attributes (still a Python dict-of-dicts -- not in scope for CSR).
        attr_nids   = z['attr_nids']
        attr_names  = z['attr_names']
        attr_values = z['attr_values']
        for nid, name, val in zip(attr_nids.tolist(),
                                  attr_names.tolist(),
                                  attr_values.tolist()):
            g.node_attrs[nid][name] = val

        return g

    # Accessors

    def get_attr(self, node_id: int, attr: str,
                 default: Optional[str] = None) -> Optional[str]:
        """Return the value of *attr* for *node_id*, or *default* if absent."""
        return self.node_attrs.get(node_id, {}).get(attr, default)

    def out_degree(self, node_id: int) -> int:
        # CSR fast path: avoids constructing a list just to take its length.
        offs = getattr(self, "_out_offsets", None)
        if offs is not None and 0 <= node_id < self.num_vertices:
            return int(offs[node_id + 1] - offs[node_id])
        return len(self.out_adj.get(node_id, []))

    def in_degree(self, node_id: int) -> int:
        offs = getattr(self, "_in_offsets", None)
        if offs is not None and 0 <= node_id < self.num_vertices:
            return int(offs[node_id + 1] - offs[node_id])
        return len(self.in_adj.get(node_id, []))

    def get_type(self, node_id: int) -> Optional[int]:
        return self.node_type.get(node_id)

    def type_name(self, node_id: int) -> str:
        t = self.node_type.get(node_id)
        return self.inv_node_type.get(t, "?") if t is not None else "?"

    def edge_name(self, label_id: int) -> str:
        return self.inv_edge_label.get(label_id, str(label_id))

    def nodes_by_type(self, type_id: int) -> List[int]:
        """Return all node ids whose type equals *type_id*."""
        arr = getattr(self, "_node_type_array", None)
        if arr is not None:
            import numpy as np
            return np.where(arr == type_id)[0].tolist()
        return [nid for nid, t in self.node_type.items() if t == type_id]

    def has_edge(self, src: int, dst: int,
                 label: Optional[int] = None) -> bool:
        """Return True if a directed edge src→dst exists (optionally filtered by *label*)."""
        offs = getattr(self, "_out_offsets", None)
        if offs is not None:
            if not (0 <= src < self.num_vertices):
                return False
            s = int(offs[src])
            e = int(offs[src + 1])
            if s == e:
                return False
            dsts_slice = self._out_dsts[s:e]
            mask = (dsts_slice == dst)
            if not mask.any():
                return False
            if label is None:
                return True
            return bool((mask & (self._out_lbls[s:e] == label)).any())
        # Fallback path (graph not yet compacted).
        for d, l in self.out_adj.get(src, []):
            if d == dst:
                if label is None or l == label:
                    return True
        return False

    def has_out_label(self, src: int, label: int) -> bool:
        """Return True if `src` has ANY out-edge with `label` in THIS (base) graph."""
        offs = getattr(self, "_out_offsets", None)
        if offs is not None:
            if not (0 <= src < self.num_vertices):
                return False
            s = int(offs[src]); e = int(offs[src + 1])
            if s == e:
                return False
            return bool((self._out_lbls[s:e] == label).any())
        for _d, l in self.out_adj.get(src, []):
            if l == label:
                return True
        return False

    def has_in_edge(self, dst: int, src: int,
                    label: Optional[int] = None) -> bool:
        """Return True if a directed edge src→dst exists, checking via in_adj."""
        return self.has_edge(src, dst, label)

    # Statistics

    def summary(self, verbose: bool = False) -> str:
        type_counts = defaultdict(int)
        for t in self.node_type.values():
            type_counts[t] += 1
        label_counts: Dict[int, int] = defaultdict(int)
        for edges in self.out_adj.values():
            for _, l in edges:
                label_counts[l] += 1

        lines = [
            f"Graph summary",
            f"  vertices : {self.num_vertices:,}  ({len(type_counts)} types)",
            f"  edges    : {self.num_edges:,}  ({len(label_counts)} labels)",
        ]
        if verbose:
            lines.append("  node types:")
            for tid, count in sorted(type_counts.items()):
                name = self.inv_node_type.get(tid, str(tid))
                lines.append(f"    [{tid}] {name:20s}  {count:,}")
            lines.append("  edge labels:")
            for lid, count in sorted(label_counts.items()):
                name = self.inv_edge_label.get(lid, str(lid))
                lines.append(f"    [{lid}] {name:20s}  {count:,}")
        return "\n".join(lines)
