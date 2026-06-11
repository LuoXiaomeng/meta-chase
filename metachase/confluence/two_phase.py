"""Confluence — V.C two-phase protocol (paper Section V.C)."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..chase.coordinator import VanillaCoordinator
from ..graph.derived import DerivedFacts
from ..graph.gar import GAR
from .canon import _candidates_by_key, decide, gar_head_label

Triple = Tuple[int, int, int]


# Rule-structure helpers

def _body_labels(gar: GAR) -> Set[int]:
    """Edge-label ids this GAR CONSUMES in its body pattern (wildcards dropped)."""
    return {e.label for e in gar.pattern.edges if e.label is not None}


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


def _dependency_strata(gars: List[GAR]) -> List[List[int]]:
    """Partition rule indices into dependency STRATA, returned dependencies-first."""
    n = len(gars)
    heads  = [gar_head_label(g) for g in gars]
    bodies = [_body_labels(g) for g in gars]
    adj: List[List[int]] = [[] for _ in range(n)]      # i -> j : i before j
    for i in range(n):
        h = heads[i]
        if h is None:
            continue
        for j in range(n):
            if h in bodies[j]:
                adj[i].append(j)

    # Iterative Tarjan SCC -> components in REVERSE topological order.
    idx_counter = [0]
    stack: List[int] = []
    on_stack = [False] * n
    index   = [-1] * n
    lowlink = [0] * n
    sccs: List[List[int]] = []

    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = lowlink[v] = idx_counter[0]
                idx_counter[0] += 1
                stack.append(v); on_stack[v] = True
            recursed = False
            for k in range(pi, len(adj[v])):
                w = adj[v][k]
                if index[w] == -1:
                    work[-1] = (v, k + 1)
                    work.append((w, 0))
                    recursed = True
                    break
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], index[w])
            if recursed:
                continue
            if lowlink[v] == index[v]:                 # v is an SCC root
                comp = []
                while True:
                    w = stack.pop(); on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                sccs.append(comp)
            work.pop()
            if work:                                   # propagate to parent
                u = work[-1][0]
                lowlink[u] = min(lowlink[u], lowlink[v])

    sccs.reverse()                                     # -> dependencies first
    return sccs


def _evidence_consumes_p_out(evid: List[GAR], p_out_ids: Set[int]) -> bool:
    """True iff some EVIDENCE rule consumes a P_out predicate in its body."""
    return any(p_out_ids & _body_labels(g) for g in evid)


def _collect_p_out_from_derived(derived, p_out_ids):
    """Extract P_out (src, label, dst) triples from a DerivedFacts overlay."""
    out = set()
    for (src, dst, label) in derived._edge_set:
        if label in p_out_ids:
            out.add((src, label, dst))
    return out


# V.C  Two-phase protocol

def two_phase_chase(graph, gars: List[GAR], p_out_ids: Set[int],
                    matcher=None, max_rounds: int = 15,
                    max_matches: int = 0, verbose: bool = False,
                    functional=None, pca_by_gar: "Optional[Dict[str, float]]" = None,
                    motif_order: "Optional[List[List[GAR]]]" = None,
                    max_passes: int = 20,
                    ) -> Set[Triple]:
    """Two-phase protocol (paper Sec V.C): derive evidence, defer decisions."""
    derived     = DerivedFacts()
    fired_pairs = set()
    # functional label-id set -> base authority for those keys (same as run_subset)
    func_ids = None
    if functional is not None:
        func_ids = ({lid for lid, v in functional.items() if v}
                    if isinstance(functional, dict) else set(functional))

    all_results = []
    provenance: Dict[Triple, set] = {}     # merged P_out producers

    def _run(group, tag):
        """Run one rule group over the shared overlay; accumulate results + provenance."""
        if not group:
            return
        c = VanillaCoordinator(
            graph, group, max_rounds=max_rounds, max_matches=max_matches,
            verbose=verbose, matcher=matcher, p_out_ids=p_out_ids,
            functional_p_out=func_ids)   # two-phase never early-commits (= forced unified decision)
        all_results.extend(
            c.run(existing_derived=derived, existing_fired_pairs=fired_pairs, task_id=tag))
        for triple, gids in c.p_out_provenance.items():
            provenance.setdefault(triple, set()).update(gids)

    if motif_order is not None:
        # (1) POLICY-ORDER (仅调整 motif 顺序): run WHOLE motifs in policy order with the
        evid_motifs = [m for m in motif_order
                       if not any(gar_head_label(g) in p_out_ids for g in m)]
        dec_motifs  = [m for m in motif_order
                       if any(gar_head_label(g) in p_out_ids for g in m)]
        for mi, m in enumerate(evid_motifs):
            _run(m, f"E{mi}")
        for mi, m in enumerate(dec_motifs):
            _run(m, f"D{mi}")
    else:
        # (2) DEPENDENCY-STRATA (+ gate to paper's binary split when P_out terminal).
        evid, dec = split_evidence_decision(gars, p_out_ids)
        strata_gars = ([[gars[i] for i in s] for s in _dependency_strata(gars)]
                       if _evidence_consumes_p_out(evid, p_out_ids) else [evid, dec])
        for si, sg in enumerate(strata_gars):
            _run(sg, f"S{si}")

    # Resolve competing candidates via the shared DECISION LOGIC (max PCA over ALL
    if pca_by_gar is not None:
        return decide(_candidates_by_key(all_results, p_out_ids, pca_by_gar,
                                         provenance), functional)
    return _collect_p_out_from_derived(derived, p_out_ids)
