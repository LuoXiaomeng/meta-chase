"""Load AMIE rules from a rules.json file and produce GAR objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..graph.gar import (
    GAR, Conclusion, ConclusionType,
    PatternEdge, PatternGraph, PatternNode,
)

ROOT         = Path(__file__).resolve().parents[2]
RULES_JSON   = ROOT / "tools" / "amie" / "rules.json"
EDGE_DICT_JSON = ROOT / "data" / "dbpedia" / "processed" / "edge_label_dict.json"

# Fallback mappings for predicates that keep namespace prefixes
_PRED_ALIASES: Dict[str, str] = {
    "http://xmlns.com/foaf/0.1/logo":      "logo",
    "http://xmlns.com/foaf/0.1/depiction": "depiction",
    "rdfs:seeAlso":                        "seeAlso",
    "owl:differentFrom":                   "differentFrom",
}


# Internal helpers

def _load_edge_dict(edge_dict_path=None) -> Dict[str, int]:
    path = Path(edge_dict_path) if edge_dict_path is not None else EDGE_DICT_JSON
    raw = json.loads(path.read_text())
    # Add alias mappings so namespace-prefixed predicates resolve too
    result = dict(raw)
    for full, short in _PRED_ALIASES.items():
        if short in raw:
            result[full] = raw[short]
    return result


def _rule_to_gar(rule: dict, edge_dict: Dict[str, int]) -> Optional[GAR]:
    """Convert one JSON rule record to a GAR."""
    nodes: Dict[str, PatternNode] = {}
    edges: List[PatternEdge]       = []

    # Body atoms → PatternNodes + PatternEdges
    for subj_var, pred, obj_var in rule["body_atoms"]:
        if pred not in edge_dict:
            return None
        for v in (subj_var, obj_var):
            if v not in nodes:
                nodes[v] = PatternNode(var=v, label=None)   # type wildcard
        edges.append(PatternEdge(
            src_var=subj_var,
            dst_var=obj_var,
            label=edge_dict[pred],
        ))

    # Head atom
    h_subj, h_pred, h_obj = rule["head_atom"]
    if h_pred not in edge_dict:
        return None
    for v in (h_subj, h_obj):
        if v not in nodes:
            nodes[v] = PatternNode(var=v, label=None)

    body_preds  = [atom[1] for atom in rule["body_atoms"]]
    head_pred   = rule["head_atom"][1]
    pca         = rule["pca"]
    body_pred_ids = [edge_dict[p] for p in body_preds]

    description = (
        f"AMIE  {'  ∧  '.join(body_preds)}  ⟹  {head_pred}"
        f"  [pca={pca:.2f} sup={rule['support']} conf={pca:.2f}]"
    )

    return GAR(
        id=rule["id"],
        pattern=PatternGraph(nodes=nodes, edges=edges),
        predicates=[],           # guards handled externally via body_pred_ids
        conclusion=Conclusion.enrich_edge(
            src_var=h_subj,
            edge_label_id=edge_dict[h_pred],
            dst_var=h_obj,
            label=f"{head_pred}_candidate",
        ),
        description=description,
    )


# Public API

def is_propagation_rule(rule: dict) -> bool:
    """True iff the rule's head predicate also appears in its body."""
    head_pred  = rule["head_atom"][1]
    body_preds = {a[1] for a in rule["body_atoms"]}
    return head_pred in body_preds


def load_amie_gars(
    rules_json:        Path  = RULES_JSON,
    path_only:         bool  = False,
    min_pca:           float = 0.0,
    min_sup:           int   = 0,
    exclude_propagation: bool = False,
    edge_dict_path:    "Path | None" = None,
) -> Tuple[List[GAR], List[dict]]:
    """Load GARs from rules.json."""
    edge_dict = _load_edge_dict(edge_dict_path)
    raw_rules = json.loads(rules_json.read_text())

    gars: List[GAR]  = []
    meta: List[dict] = []
    skipped = 0
    skipped_prop = 0

    for r in raw_rules:
        if path_only and not r.get("is_path", True):
            continue
        if r["pca"] < min_pca or r["support"] < min_sup:
            continue
        if exclude_propagation and is_propagation_rule(r):
            skipped_prop += 1
            continue
        gar = _rule_to_gar(r, edge_dict)
        if gar is None:
            skipped += 1
            continue
        gars.append(gar)
        meta.append(r)

    msg = f"[amie_loader] Loaded {len(gars)} GARs  (skipped {skipped} — predicate not in graph"
    if exclude_propagation:
        msg += f"; {skipped_prop} propagation rules excluded"
    msg += ")"
    print(msg)
    return gars, meta
