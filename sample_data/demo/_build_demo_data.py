"""Generate the bundled `demo` dataset (graph CSVs + dicts + rules.json).

Run once to (re)create the files under this directory:

    python sample_data/demo/_build_demo_data.py

The graph is tiny but exercises every chase mechanism the engine has:
  * a producer -> consumer chain  (cityOf + country  -> locationCountry),
  * a recursive (cyclic) SCC      (locationCountry propagates along parentCompany),
  * competing rules on a FUNCTIONAL P_out predicate (locationCountry has two
    producers with different PCA -> canonicalization picks the max-PCA value),
  * a second P_out predicate (owner) derived from a 2-hop pattern.

It is fully synthetic; the files it writes are committed so `demo.py` runs with
no build step.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROC = HERE / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# --- node types -----------------------------------------------------------
NODE_TYPES = {"Person": 0, "City": 1, "Country": 2, "Company": 3}

# node_id -> (name, type_name)
NODES = {
    0:  ("Alice",    "Person"),
    1:  ("Bob",      "Person"),
    2:  ("Carol",    "Person"),
    3:  ("Paris",    "City"),
    4:  ("Lyon",     "City"),
    5:  ("Berlin",   "City"),
    6:  ("France",   "Country"),
    7:  ("Germany",  "Country"),
    8:  ("AcmeFR",   "Company"),
    9:  ("AcmeDE",   "Company"),
    10: ("ParentCo", "Company"),
    11: ("SubCo",    "Company"),
}

# --- edge labels ----------------------------------------------------------
EDGE_LABELS = {
    "bornIn":          0,   # Person -> City
    "cityOf":          1,   # City -> Country        (base evidence)
    "country":         2,   # Company -> Country     (base evidence)
    "headquarteredIn": 3,   # Company -> City        (base evidence)
    "parentCompany":   4,   # Company -> Company     (base evidence)
    "locationCountry": 5,   # * -> Country           (P_out, functional)
    "owner":           6,   # Company -> Company     (P_out)
    "livesIn":         7,   # Person -> Country      (intermediate)
}

# --- base edges (src, dst, label_name) ------------------------------------
EDGES = [
    # cities -> their country
    (3, 6, "cityOf"),       # Paris  -> France
    (4, 6, "cityOf"),       # Lyon   -> France
    (5, 7, "cityOf"),       # Berlin -> Germany
    # people born in a city
    (0, 3, "bornIn"),       # Alice  -> Paris
    (1, 4, "bornIn"),       # Bob    -> Lyon
    (2, 5, "bornIn"),       # Carol  -> Berlin
    # companies
    (8, 3, "headquarteredIn"),   # AcmeFR -> Paris
    (9, 5, "headquarteredIn"),   # AcmeDE -> Berlin
    (8, 6, "country"),           # AcmeFR -> France   (agrees with its Paris HQ)
    # AcmeDE is headquartered in Berlin (-> Germany via R002, PCA 0.80) but DECLARES
    # France as its country (-> France via R003, PCA 0.99). The two P_out producers
    # CONFLICT on the functional locationCountry key; canonicalization resolves it to
    # the max-PCA value (France).
    (9, 6, "country"),           # AcmeDE -> France   (conflicts with Berlin HQ)
    (10, 11, "parentCompany"),   # ParentCo -> SubCo
    (11, 10, "parentCompany"),   # SubCo -> ParentCo   -> a 2-cycle (recursive SCC)
    (10, 6, "country"),          # ParentCo -> France
]


def build():
    inv_nodes = NODES
    # edges.csv : src,dst  (no header) ; edge_labels.csv : edge_index,label_id
    with open(PROC / "edges.csv", "w", newline="") as ef, \
         open(PROC / "edge_labels.csv", "w", newline="") as lf:
        ew = csv.writer(ef)
        lw = csv.writer(lf)
        lw.writerow(["edge_index", "label_id"])
        for i, (s, d, lab) in enumerate(EDGES):
            ew.writerow([s, d])
            lw.writerow([i, EDGE_LABELS[lab]])

    # node_labels.csv : node_id,label_id (header)
    with open(PROC / "node_labels.csv", "w", newline="") as nf:
        nw = csv.writer(nf)
        nw.writerow(["node_id", "label_id"])
        for nid, (_name, tname) in sorted(inv_nodes.items()):
            nw.writerow([nid, NODE_TYPES[tname]])

    (PROC / "node_type_dict.json").write_text(json.dumps(NODE_TYPES, indent=2))
    (PROC / "edge_label_dict.json").write_text(json.dumps(EDGE_LABELS, indent=2))
    # node id -> name, handy for pretty-printing the demo output.
    (PROC / "node_id_dict.json").write_text(
        json.dumps({str(nid): name for nid, (name, _t) in inv_nodes.items()}, indent=2))

    # --- rules.json -------------------------------------------------------
    # Rule record shape consumed by amie_loader:
    #   {id, body_atoms:[[subj,pred,obj],...], head_atom:[subj,pred,obj], pca, support}
    rules = [
        # CHAIN: a person's livesIn country = the country of the city they were born in.
        {"id": "R001",
         "body_atoms": [["x", "bornIn", "c"], ["c", "cityOf", "n"]],
         "head_atom":  ["x", "livesIn", "n"], "pca": 0.95, "support": 100},

        # P_out (locationCountry) producer #1: a company's locationCountry = the
        # country of the city it is headquartered in (via the derived/base cityOf).
        {"id": "R002",
         "body_atoms": [["co", "headquarteredIn", "c"], ["c", "cityOf", "n"]],
         "head_atom":  ["co", "locationCountry", "n"], "pca": 0.80, "support": 60},

        # P_out (locationCountry) producer #2: a company's locationCountry is its
        # declared `country` edge. Competes with R002 on the SAME functional key;
        # higher PCA -> canonicalization prefers this value.
        {"id": "R003",
         "body_atoms": [["co", "country", "n"]],
         "head_atom":  ["co", "locationCountry", "n"], "pca": 0.99, "support": 90},

        # RECURSIVE P_out rule: locationCountry propagates along parentCompany.
        # parentCompany forms a 2-cycle in the base graph -> this rule + R003 land
        # in one cyclic SCC block (recursion contained in the block).
        {"id": "R004",
         "body_atoms": [["a", "parentCompany", "b"], ["b", "locationCountry", "n"]],
         "head_atom":  ["a", "locationCountry", "n"], "pca": 0.70, "support": 40},

        # P_out (owner): the parent company owns its subsidiary.
        {"id": "R005",
         "body_atoms": [["a", "parentCompany", "b"]],
         "head_atom":  ["b", "owner", "a"], "pca": 0.85, "support": 50},
    ]
    (HERE / "rules.json").write_text(json.dumps(rules, indent=2))
    print(f"wrote demo dataset to {HERE}")
    print(f"  {len(EDGES)} edges, {len(NODES)} nodes, {len(rules)} rules")


if __name__ == "__main__":
    build()
