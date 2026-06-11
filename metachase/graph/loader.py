"""Loader for the in-memory :class:`~metachase.graph.graph.Graph`."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

from .graph import Graph


def load_processed(processed_dir: str | Path, use_cache: bool = True) -> Graph:
    """Load a :class:`Graph` from a *processed/* directory."""
    import time as _time
    d = Path(processed_dir)

    # Try binary cache first -- 20-50x faster than re-parsing CSVs.
    cache_path = d / "_graph_cache_v3.npz"
    if use_cache and cache_path.exists():
        try:
            csv_mtimes = [
                (d / fn).stat().st_mtime
                for fn in ('edges.csv', 'edge_labels.csv',
                           'node_labels.csv', 'node_attrs.csv')
                if (d / fn).exists()
            ]
            if csv_mtimes and cache_path.stat().st_mtime >= max(csv_mtimes):
                t = _time.perf_counter()
                g = Graph._from_npz_cache(cache_path, d)
                print(f"[Graph] loaded from cache in {_time.perf_counter()-t:.2f}s")
                return g
        except Exception as e:
            print(f"[Graph] cache load failed ({e}), rebuilding from CSV ...")

    g = Graph()

    # --- label dictionaries -------------------------------------------
    g.node_type_dict  = json.loads((d / "node_type_dict.json").read_text())
    g.edge_label_dict = json.loads((d / "edge_label_dict.json").read_text())
    g.inv_node_type   = {v: k for k, v in g.node_type_dict.items()}
    g.inv_edge_label  = {v: k for k, v in g.edge_label_dict.items()}

    # --- node types ---------------------------------------------------
    try:
        import numpy as np
        arr = np.loadtxt(d / "node_labels.csv", delimiter=',',
                         skiprows=1, dtype=np.int64)
        if arr.ndim == 1:  # single row edge case
            arr = arr.reshape(1, -1)
        for nid, lid in zip(arr[:, 0].tolist(), arr[:, 1].tolist()):
            g.node_type[nid] = lid
    except Exception:
        with open(d / "node_labels.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nid  = int(row["node_id"])
                lid  = int(row["label_id"])
                g.node_type[nid] = lid

    g.num_vertices = len(g.node_type)

    # --- edge labels + edges (read together as numpy, then build dicts) -
    try:
        import numpy as np
        elabel_arr = np.loadtxt(d / "edge_labels.csv", delimiter=',',
                                skiprows=1, dtype=np.int32, usecols=(1,))
        if elabel_arr.ndim == 0:
            elabel_arr = elabel_arr.reshape(1)
        edge_labels: List[int] = elabel_arr.tolist()

        edges_arr = np.loadtxt(d / "edges.csv", delimiter=',', dtype=np.int64)
        if edges_arr.ndim == 1:
            edges_arr = edges_arr.reshape(1, -1)
        srcs_list = edges_arr[:, 0].tolist()
        dsts_list = edges_arr[:, 1].tolist()
        for src, dst, lbl in zip(srcs_list, dsts_list, edge_labels):
            g.out_adj[src].append((dst, lbl))
            g.in_adj[dst].append((src, lbl))
    except Exception:
        edge_labels = []
        with open(d / "edge_labels.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                edge_labels.append(int(row["label_id"]))
        with open(d / "edges.csv") as f:
            reader = csv.reader(f)
            for eid, row in enumerate(reader):
                src, dst = int(row[0]), int(row[1])
                lbl = edge_labels[eid] if eid < len(edge_labels) else 0
                g.out_adj[src].append((dst, lbl))
                g.in_adj[dst].append((src, lbl))

    g.num_edges = len(edge_labels)

    # --- node attributes (optional) -----------------------------------
    attrs_path = d / "node_attrs.csv"
    if attrs_path.exists():
        # Try pandas for speed (vectorized parsing), then fall back to csv.
        try:
            import pandas as pd
            df = pd.read_csv(attrs_path, dtype={'node_id': 'int64',
                                               'attr': 'string',
                                               'value': 'string'},
                             keep_default_na=False)
            nids   = df['node_id'].tolist()
            attrs  = df['attr'].tolist()
            values = df['value'].tolist()
            for nid, attr, value in zip(nids, attrs, values):
                g.node_attrs[nid][attr] = value
        except Exception:
            with open(attrs_path, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    nid   = int(row["node_id"])
                    attr  = row["attr"]
                    value = row["value"]
                    g.node_attrs[nid][attr] = value
        print(
            f"[Graph] loaded  {sum(len(v) for v in g.node_attrs.values()):,} "
            f"attribute values for {len(g.node_attrs):,} nodes"
        )
    else:
        print("[Graph] no node_attrs.csv found — attribute predicates disabled")

    # Sanity check: number of distinct edge labels should be O(|edge_label_dict|),
    unique_labels = len(set(edge_labels))
    expected_max  = max(len(g.edge_label_dict) * 2, 1024)
    if unique_labels > expected_max:
        raise ValueError(
            f"[Graph] sanity check failed: parsed {unique_labels} distinct "
            f"edge labels, but edge_label_dict only has {len(g.edge_label_dict)}. "
            f"Likely cause: edge_labels.csv column was read wrong "
            f"(expected label_id column, got edge_index). "
            f"Delete cache and verify CSV column order."
        )

    print(
        f"[Graph] loaded  {g.num_vertices:,} vertices, "
        f"{g.num_edges:,} edges  ({processed_dir})"
    )

    # Migrate from dict-of-list to CSR-backed proxies.
    t_csr = _time.perf_counter()
    g._compact_to_csr()
    print(f"[Graph] compacted to CSR in {_time.perf_counter()-t_csr:.2f}s")

    # Write binary cache for next time (best-effort, never fatal).
    if use_cache:
        try:
            t = _time.perf_counter()
            g._save_npz_cache(cache_path)
            print(f"[Graph] wrote cache to {cache_path.name} "
                  f"({_time.perf_counter()-t:.1f}s)")
        except Exception as e:
            print(f"[Graph] cache write failed: {e}")

    return g


# Attach as a classmethod so existing ``Graph.from_processed(...)`` callers keep
def _from_processed(cls, processed_dir: str | Path, use_cache: bool = True) -> Graph:
    return load_processed(processed_dir, use_cache=use_cache)


Graph.from_processed = classmethod(_from_processed)
