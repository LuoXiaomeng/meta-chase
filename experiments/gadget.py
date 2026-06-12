"""gadget — synthesize a conflict-injected copy of the base graph for e4_3 / e4_4 (plants
conflict units at four SCC boundaries; only the graph changes, the rules do not). Pure file I/O.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

NEEDED = ("owningOrganisation", "owningCompany", "owner", "country", "locationCountry",
          "location", "headquarter", "nationality", "stateOfOrigin",
          "successor", "predecessor", "ground", "homeStadium")

BRIDGE = {"owner": "owningCompany", "headquarter": "headquarter",
          "nationality": "nationality", "predecessor": "predecessor"}


def _unit_owner(add, alloc, L, F):
    a, h1, h2 = alloc(), alloc(), alloc()
    add(a, h1, L["owningOrganisation"]); add(a, h2, L["owningOrganisation"])
    add(h1, alloc(), L["country"])
    for _ in range(F):
        add(h2, alloc(), L["country"])


def _unit_headquarter(add, alloc, L, F):
    a, h1, h2 = alloc(), alloc(), alloc()
    add(a, h1, L["location"]); add(a, h2, L["location"])
    add(h1, alloc(), L["country"])
    for _ in range(F):
        add(h2, alloc(), L["country"])


def _unit_nationality(add, alloc, L, F):
    a, b1, b2 = alloc(), alloc(), alloc()
    add(a, b1, L["country"]); add(a, b2, L["country"])


def _unit_predecessor(add, alloc, L, F):
    a, h1, h2 = alloc(), alloc(), alloc()
    add(h1, a, L["successor"]); add(h2, a, L["successor"])
    add(h1, alloc(), L["ground"])
    for _ in range(F):
        add(h2, alloc(), L["ground"])


BUILDERS = {"owner": _unit_owner, "headquarter": _unit_headquarter,
            "nationality": _unit_nationality, "predecessor": _unit_predecessor}


def inject(per_boundary, name, base_graph_dir, out_root, fanout=20,
           boundaries=("owner", "headquarter", "nationality", "predecessor")):
    """Write a conflict-injected processed graph to out_root/<name>/processed/."""
    base = Path(base_graph_dir)
    eld = json.loads((base / "edge_label_dict.json").read_text())
    L = {k: eld[k] for k in NEEDED}

    mx = 0
    with open(base / "node_labels.csv") as f:
        next(f)
        for row in f:
            if row.strip():
                mx = max(mx, int(row.split(",")[0]))
    nid = [mx + 1]

    out = Path(out_root) / name / "processed"
    out.mkdir(parents=True, exist_ok=True)
    for fn in ("edge_label_dict.json", "node_type_dict.json"):
        shutil.copyfile(base / fn, out / fn)

    base_edges = (base / "edges.csv").read_text().splitlines()
    base_elabels = [int(r.split(",")[1]) for r in
                    (base / "edge_labels.csv").read_text().splitlines()[1:]]
    base_nlabels = (base / "node_labels.csv").read_text().splitlines()[1:]

    inj_edges, inj_labels, inj_nodes = [], [], []

    def add(s, d, l):
        inj_edges.append((s, d)); inj_labels.append(l)

    def alloc():
        v = nid[0]; nid[0] += 1; inj_nodes.append(v); return v

    for bname in boundaries:
        for _ in range(per_boundary):
            BUILDERS[bname](add, alloc, L, fanout)

    with open(out / "edges.csv", "w") as fe:
        fe.write("\n".join(base_edges))
        if base_edges and not base_edges[-1].endswith("\n"):
            fe.write("\n")
        for s, d in inj_edges:
            fe.write(f"{s},{d}\n")
    with open(out / "edge_labels.csv", "w") as fl:
        fl.write("edge_id,label_id\n")
        eid = 0
        for lb in base_elabels:
            fl.write(f"{eid},{lb}\n"); eid += 1
        for lb in inj_labels:
            fl.write(f"{eid},{lb}\n"); eid += 1
    with open(out / "node_labels.csv", "w") as fn:
        fn.write("node_id,label_id\n")
        for row in base_nlabels:
            fn.write(row + "\n")
        for ndid in inj_nodes:
            fn.write(f"{ndid},0\n")
    cache = out / "_graph_cache_v3.npz"
    if cache.exists():
        cache.unlink()

    bridges = sorted({eld[BRIDGE[b]] for b in boundaries})
    bridge_names = sorted({BRIDGE[b] for b in boundaries})
    (out / "bridges.json").write_text(json.dumps(
        {"boundaries": list(boundaries), "bridge_label_ids": bridges,
         "bridge_names": bridge_names, "per_boundary": per_boundary,
         "fanout": fanout}, indent=2))
    return {"name": name, "graph_dir": out, "bridge_label_ids": bridges,
            "bridge_names": bridge_names}
