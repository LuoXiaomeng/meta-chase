"""session.py -- the shared `Context` for the Meta-Chase / AFF engine."""

from __future__ import annotations

from .chase.coordinator import VanillaCoordinator
from .graph.derived import DerivedFacts
from .graph.graph import Graph
from .graph.loader import load_processed   # attaches Graph.from_processed
from .match.matcher import Matcher
from .rules.amie_loader import load_amie_gars
from .rules.group_rules import (
    compute_scc_blocks, relevant_rules)
from .rules.dataset_config import get_config
from .rules.p_out import resolve_p_out_ids


# Shared context

class Context:
    """Everything loaded once and reused by the AFF engine."""

    def __init__(self, dataset: str):
        self.cfg = get_config(dataset)
        print(f"Loading graph from {self.cfg.graph_dir} ...")
        self.graph = Graph.from_processed(self.cfg.graph_dir)
        print(self.graph.summary())

        # FULL rule set (incl. propagation) -- the classical-FF baseline.
        gars_full, meta_full = load_amie_gars(
            self.cfg.rules_json, path_only=self.cfg.path_only,
            min_pca=self.cfg.min_pca, min_sup=self.cfg.min_sup,
            exclude_propagation=False, edge_dict_path=self.cfg.edge_dict_path,
        )
        self.gars_full, self.meta_full = gars_full, meta_full

        # Meta-Chase rule set = the P_out-RELEVANT subset (backward reachable from
        rel_ids = {m["id"] for m in relevant_rules(meta_full, set(self.cfg.p_out))}
        focused_meta = [m for m in meta_full if m["id"] in rel_ids]
        self.gars_focused = [g for g in gars_full if g.id in rel_ids]
        self.gars, self.meta = self.gars_focused, focused_meta

        # SCC blocks over the P_out-relevant set: cycles contained in blocks,
        blocks, self.block_dag = compute_scc_blocks(focused_meta)
        gar_by_id  = {g.id: g for g in gars_full}
        meta_by_id = {m["id"]: m for m in meta_full}
        self.clusters_gars = [[gar_by_id[r["id"]] for r in c if r["id"] in gar_by_id]
                              for c in blocks]
        self.clusters_meta = [[meta_by_id[r["id"]] for r in c if r["id"] in meta_by_id]
                              for c in blocks]
        self.n_tasks = len(self.clusters_gars)

        self.p_out_ids  = resolve_p_out_ids(self.cfg.p_out, self.graph.edge_label_dict)
        self.pca_by_gar = {m["id"]: m.get("pca", 0.0) for m in focused_meta}
        self.matcher    = Matcher(self.graph, max_matches=0)
        # Per-P_out-predicate FUNCTIONALITY:
        if getattr(self.cfg, "functional_override", None) is not None:
            override_ids = resolve_p_out_ids(self.cfg.functional_override,
                                             self.graph.edge_label_dict)
            self.functional_p_out = {lid: (lid in override_ids)
                                     for lid in self.p_out_ids}
        else:
            self.functional_p_out = _p_out_functionality(self.cfg.graph_dir, self.p_out_ids)
        # Set of FUNCTIONAL P_out label ids -> base graph is authoritative for them
        self.functional_p_out_ids = {lid for lid, v in self.functional_p_out.items() if v}
        n_cyc = sum(1 for b in blocks if len(b) > 1)
        n_func = sum(1 for v in self.functional_p_out.values() if v)
        print(f"full ruleset: {len(gars_full)};  Meta-Chase (P_out-relevant): "
              f"{len(self.gars_focused)} rules -> {self.n_tasks} SCC blocks "
              f"({n_cyc} cyclic) + {len(self.block_dag)} DAG edges;  "
              f"P_out={sorted(self.cfg.p_out)} ({len(self.p_out_ids)} resolved, "
              f"{n_func} functional)")

    # --- small reusable runner ---------------------------------------------

    def run_subset(self, gars, derived=None, fired=None, max_rounds=0,
                   max_derived=0, task_id="X", base_authority=True):
        """Chase one rule subset (optionally on a shared IRS)."""
        derived = DerivedFacts() if derived is None else derived
        fired = set() if fired is None else fired
        coord = VanillaCoordinator(
            self.graph, gars, max_rounds=max_rounds, max_matches=0,
            max_derived=max_derived, verbose=False, matcher=self.matcher,
            p_out_ids=self.p_out_ids,
            functional_p_out=(self.functional_p_out_ids if base_authority else None))
        results = coord.run(existing_derived=derived,
                            existing_fired_pairs=fired, task_id=task_id)
        return coord, results


def task_aware_run(ctx: Context, order, max_rounds, max_matches=0,
                   to_quiescence=False, max_passes=6, early_commit=False):
    """Run SCC blocks in `order` sharing one IRS; return (p_out, results, per_task, prov)."""
    from .aff import AFFConfig, AFFRunner
    afr = AFFRunner(ctx, AFFConfig(
        task_source="scc_blocks",
        order_strategy="custom", custom_order=list(order),
        sharing="shared",
        decision_mode=("early_commit" if early_commit else "monotone"),
        to_quiescence=to_quiescence,
        max_passes=max_passes,
        max_rounds=max_rounds,
        max_matches=max_matches,
        base_authority=True,
    )).run()
    return afr.p_out, afr.all_results, afr.per_task, afr.provenance


def _p_out_functionality(graph_dir, p_out_ids, thresh=0.9):
    """Classify each P_out predicate functional (single-valued) vs set-valued from the BASE graph: functionality = #distinct-subjects / #edges (~1 => functional)."""
    import csv as _csv
    from collections import defaultdict
    from pathlib import Path as _Path
    pdir = _Path(graph_dir)
    pset = set(p_out_ids)
    subj = defaultdict(set)   # label_id -> set of subjects
    cnt = defaultdict(int)    # label_id -> #edges
    try:
        with open(pdir / "edges.csv") as ef, open(pdir / "edge_labels.csv") as lf:
            next(lf, None)
            for er, lr in zip(_csv.reader(ef), _csv.reader(lf)):
                if not er or not lr:
                    continue
                lid = int(lr[1] if len(lr) > 1 else lr[0])
                if lid in pset:
                    subj[lid].add(int(er[0])); cnt[lid] += 1
    except FileNotFoundError:
        return {lid: True for lid in pset}
    out = {}
    for lid in pset:
        if cnt[lid] == 0:
            out[lid] = True                       # derived-only -> default functional
        else:
            out[lid] = (len(subj[lid]) / cnt[lid]) >= thresh
    return out
