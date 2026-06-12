"""Confluence — V.C two-phase protocol (paper Section V.C)."""

from __future__ import annotations

from typing import List, Set, Tuple

from ..graph.gar import GAR
from .canon import gar_head_label

Triple = Tuple[int, int, int]


def split_evidence_decision(gars: List[GAR], p_out_ids: Set[int]
                            ) -> Tuple[List[GAR], List[GAR]]:
    """Partition rules into (Sigma_evid, Sigma_dec)."""
    evid, dec = [], []
    for g in gars:
        if gar_head_label(g) in p_out_ids:
            dec.append(g)
        else:
            evid.append(g)
    return evid, dec


def two_phase_alternate(evid, dec, derived, run_group, inc_ok,
                        seed_edges=None, tag=""):
    """SINGLE SOURCE OF TRUTH for the within-task two-phase (Fig-4 TwoPhaseChase) alternation control flow."""
    reinvoke = bool(seed_edges) and inc_ok
    if reinvoke:
        ev_e, _ev_a = run_group(evid, tag + ":E",
                                seed_edges=seed_edges, incremental=True)
        dec_e, dec_a = run_group(dec, tag + ":D",
                                 seed_edges=list(seed_edges) + list(ev_e),
                                 incremental=True)
    else:
        run_group(evid, tag + ":E")
        dec_e, dec_a = run_group(dec, tag + ":D")
    while True:
        before = derived.num_edges + derived.num_attrs
        ev_e, ev_a = run_group(evid, tag + ":E",
                               seed_edges=dec_e if inc_ok else None,
                               seed_attrs=dec_a if inc_ok else None, incremental=inc_ok)
        dec_e, dec_a = run_group(dec, tag + ":D",
                                 seed_edges=ev_e if inc_ok else None,
                                 seed_attrs=ev_a if inc_ok else None, incremental=inc_ok)
        if derived.num_edges + derived.num_attrs == before:
            break


def _collect_p_out_from_derived(derived, p_out_ids):
    """Extract P_out (src, label, dst) triples from a DerivedFacts overlay."""
    out = set()
    for (src, dst, label) in derived._edge_set:
        if label in p_out_ids:
            out.add((src, label, dst))
    return out
