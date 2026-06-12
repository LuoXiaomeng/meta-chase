"""enforcement — conflict-injection + tp-canon streaming shared by e4_3 / e4_4. Injected
datasets are regenerated on demand under `_generated/` from the bundled graph; `stream_anytime`
streams SCC blocks under canon / two-phase / tp-canon (tp-canon collapses the overlay at
block boundaries once functional-P_out inconsistency exceeds a threshold).
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path

from . import datasets, gadget

CACHE = Path(__file__).resolve().parent / "_generated"
BASE_DS = "dbpedia_gamma"                           # m200, unified with e1-e3
RULES = datasets.GAMMA / "rules.json"              # the m200 rule set (227 rules)
BRIDGE_NAMES = ["owningCompany", "headquarter", "nationality", "predecessor"]
_P_OUT = {"director", "homeStadium", "locationCountry", "owner", "stateOfOrigin"}


def _cfg(name, graph_dir, rules_json=None):
    from metachase.rules.dataset_config import DatasetConfig
    return DatasetConfig(
        name=name, graph_dir=graph_dir, rules_json=rules_json or RULES,
        edge_dict_path=graph_dir / "edge_label_dict.json", p_out=set(_P_OUT),
        min_sup=1, min_pca=0.0, path_only=False, exclude_propagation=True,
        cycle_breaker_rules=[], gt_max_rounds=15, gt_max_derived=0,
        functional_override=None)


def ensure_multi(per_boundary, fanout):
    """Name of the injected dataset (generate into _generated/ + register if absent)."""
    datasets.register_paper_datasets()
    from metachase.rules.dataset_config import register, list_datasets
    if per_boundary == 0:
        return BASE_DS                              # registered by register_paper_datasets
    name = f"dbpedia_multi_n{per_boundary}_f{fanout}"
    proc = CACHE / name / "processed"
    if not (proc / "edges.csv").exists():
        with contextlib.redirect_stdout(io.StringIO()):
            gadget.inject(per_boundary, name,
                          base_graph_dir=datasets.GAMMA / "processed",
                          out_root=CACHE, fanout=fanout)
    if name not in set(list_datasets()):
        register(_cfg(name, proc))
    return name


def shuffled_multi(per_boundary, fanout, seed):
    """Injected dataset (same graph) with the m200 rule ORDER shuffled by `seed`.
    The threshold collapse in e4_4 is rule-order sensitive (it changes the focused
    SCC-block order); averaging the runtime over several shuffles gives the
    order-robust curve. seed=None returns the un-shuffled base dataset."""
    name = ensure_multi(per_boundary, fanout)
    if seed is None:
        return name
    import json, random
    from metachase.rules.dataset_config import register, list_datasets, get_config
    sname = f"{name}_sh{seed}"
    if sname not in set(list_datasets()):
        rules = json.loads(RULES.read_text())
        random.Random(seed).shuffle(rules)
        rp = CACHE / f"{sname}.rules.json"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(rules))
        register(_cfg(sname, get_config(name).graph_dir, rules_json=rp))
    return sname


def bridge_ids(ctx):
    """The four bridge edge-label ids to add to the collapse set."""
    eld = json.loads((Path(ctx.cfg.graph_dir) / "edge_label_dict.json").read_text())
    return tuple(eld[b] for b in BRIDGE_NAMES if b in eld)


def _topo_order(ctx):
    from metachase.policy import topo_order
    n = len(ctx.clusters_gars)
    return topo_order(list(range(n)), ctx.block_dag, {b: 0 for b in range(n)})


def collapse_inplace(ctx, derived, prov, extra_labels=()):
    """In-place canon collapse: for each functional P_out key (+ each `extra_labels`
    bridge predicate) with >1 value, keep the max-PCA winner and remove the losers."""
    targets = set(ctx.functional_p_out_ids) | set(extra_labels)
    bykey = defaultdict(list)
    for (s, d, l) in derived._edge_set:
        if l in targets:
            bykey[(s, l)].append(d)
    for (s, l), objs in bykey.items():
        if len(objs) <= 1:
            continue
        win = max(objs, key=lambda o: (max(
            (ctx.pca_by_gar.get(g, 0.0) for g in prov.get((s, l, o), ())),
            default=0.0), -o))
        for o in objs:
            if o != win:
                derived.remove_edge(s, o, l)


def _pr(pub, gt):
    if not pub or not gt:
        return 0.0, 0.0
    tp = len(pub & gt)
    return tp / len(pub), tp / len(gt)


def stream_anytime(ctx, policy, collapse_extra=(), threshold=1000):
    """Stream SCC blocks in topo order on a shared overlay; the policies differ only in
    how decisions are handled (canon / two-phase / tp-canon). Returns cumulative
    (steps, time) trajectories + the published P_out."""
    from metachase.graph.derived import DerivedFacts
    from metachase import confluence
    from metachase.confluence import split_evidence_decision
    from metachase.confluence.two_phase import two_phase_alternate
    from metachase.chase.coordinator import VanillaCoordinator

    func, p_out_ids = set(ctx.functional_p_out_ids), set(ctx.p_out_ids)
    func.add(86)
    blocks = ctx.clusters_gars
    order = _topo_order(ctx)
    derived, fired = DerivedFacts(), set()
    results, prov = [], {}
    cum_s, cum_t = 0, 0.0
    steps_traj, time_traj = [0], [0.0]

    def run_group(grp, tag, seed_edges=None, seed_attrs=None, incremental=False):
        nonlocal cum_s, cum_t
        if not grp:
            return [], set()
        t0 = time.perf_counter()
        coord = VanillaCoordinator(
            ctx.graph, grp, max_rounds=0, max_matches=0, max_derived=0, verbose=False,
            matcher=ctx.matcher, p_out_ids=ctx.p_out_ids,
            functional_p_out=ctx.functional_p_out_ids)
        res = coord.run(existing_derived=derived, existing_fired_pairs=fired, task_id=tag,
                        seed_delta_edges=seed_edges, seed_delta_attrs=seed_attrs,
                        incremental=incremental)
        cum_t += time.perf_counter() - t0
        cum_s += coord.run_stats.chasing_steps
        results.extend(res)
        for tr, g in coord.p_out_provenance.items():
            prov.setdefault(tr, set()).update(g)
        return coord.produced_edges, coord.produced_attrs

    for ci in order:
        if policy in ("two-phase", "tp-canon"):
            evid, dec = split_evidence_decision(blocks[ci], p_out_ids)
            two_phase_alternate(evid, dec, derived, run_group, True, tag=f"b{ci}")
        else:
            run_group(blocks[ci], f"b{ci}")
        if policy == "tp-canon":
            m = defaultdict(set)
            for (s, d, l) in derived._edge_set:
                if l in func:
                    m[(s, l)].add(d)
            ncon = sum(len(v) for v in m.values() if len(v) > 1)
            if ncon > threshold:
                t0 = time.perf_counter()
                collapse_inplace(ctx, derived, prov, extra_labels=collapse_extra)
                cum_t += time.perf_counter() - t0
        steps_traj.append(cum_s); time_traj.append(round(cum_t, 4))

    # functional-P_out inconsistency in the overlay BEFORE the final publish canon
    # (the peak the threshold gates).
    _m = defaultdict(set)
    for (s, d, l) in derived._edge_set:
        if l in func:
            _m[(s, l)].add(d)
    raw_incons = sum(len(v) for v in _m.values() if len(v) > 1)

    t0 = time.perf_counter()
    pub = set(confluence.canonicalize(results, ctx.p_out_ids, ctx.pca_by_gar,
                                      functional=ctx.functional_p_out, provenance=prov))
    cum_t += time.perf_counter() - t0
    return {"steps": cum_s, "time": round(cum_t, 4), "pub": pub,
            "raw_incons": raw_incons}
