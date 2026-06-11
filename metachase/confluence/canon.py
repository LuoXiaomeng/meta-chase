"""Confluence — V.B canonicalization + the shared DECISION LOGIC (paper Section V.B)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

from ..graph.gar import GAR, ConclusionType

Triple = Tuple[int, int, int]


# Shared rule-introspection helper (also used by the two-phase module).

def gar_head_label(gar: GAR) -> Optional[int]:
    """Edge-label id this GAR derives (ENRICH only); None for DEDUCE/CLEAN."""
    if gar.conclusion.type == ConclusionType.ENRICH:
        return gar.conclusion.target[1]
    return None


def _is_functional(label: Optional[int], functional) -> bool:
    """True if `label` is a FUNCTIONAL (single-valued) P_out predicate."""
    if functional is None:
        return True
    if isinstance(functional, dict):
        return functional.get(label, True)
    return label in functional


def collect_p_out(results, p_out_ids: Set[int]) -> Set[Triple]:
    out = set()
    for r in results:
        o = r.output
        if isinstance(o, tuple) and len(o) == 3 and o[1] in p_out_ids:
            out.add(o)
    return out


# Decision logic  --  the schedule-independent SELECTION RULE (paper Sec V.B).

def _candidates_by_key(results, p_out_ids: Set[int],
                       pca_by_gar: Dict[str, float], provenance=None):
    """Group derived P_out facts into {(subject, label): [(object, weight)]}."""
    cand: "defaultdict[Tuple[int,int], dict]" = defaultdict(dict)   # (s,l)->{d:weight}
    for r in results:
        o = r.output
        if isinstance(o, tuple) and len(o) == 3 and o[1] in p_out_ids:
            s, l, d = o
            if provenance:
                producers = provenance.get(o)
                w = (max(pca_by_gar.get(g, 0.0) for g in producers) if producers
                     else pca_by_gar.get(r.gar_id, 0.0))
            else:
                w = pca_by_gar.get(r.gar_id, 0.0)
            prev = cand[(s, l)].get(d)
            if prev is None or w > prev:
                cand[(s, l)][d] = w
    return {k: [(d, w) for d, w in dv.items()] for k, dv in cand.items()}


def decide(candidates, functional=None) -> Set[Triple]:
    """DECISION LOGIC: resolve competing candidate outputs to the published set."""
    out: Set[Triple] = set()
    for (s, l), cands in candidates.items():
        if _is_functional(l, functional):
            out.add((s, l, max(cands, key=lambda c: (c[1], -c[0]))[0]))
        else:
            out.update((s, l, d) for d, _ in cands)
    return out


# V.B  Canonicalization

def canonicalize(results, p_out_ids: Set[int],
                 pca_by_gar: Dict[str, float], functional=None,
                 provenance=None) -> Set[Triple]:
    """Resolve competing P_out candidates to a schedule-invariant output."""
    return decide(_candidates_by_key(results, p_out_ids, pca_by_gar, provenance),
                  functional)
