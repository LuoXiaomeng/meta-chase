"""adaptive — recall vs runtime under task switching, comparing AFF's state-adaptive order
against random and fixed orders.

    python -m experiments.showcase.adaptive
"""

from __future__ import annotations

import json

from experiments import datasets, curves
from experiments.experiment import (load_ctx, ground_truth_and_total, _quiet,
                                    DATA, FIGS)
from experiments import plot as P
from metachase import AFFConfig, AFFRunner
from metachase.rules.group_rules import compute_scc_blocks
from metachase.policy import (build_task_infos, task_dag_from_meta, topo_order,
                              RewardGuidedPolicy)

datasets.register_paper_datasets()

from pathlib import Path as _Path
_SHOWCASE = _Path(__file__).resolve().parent
DATA = _SHOWCASE / "data"            # redirect results into showcase/
FIGS = _SHOWCASE / "figures"

SLUG = "adaptive"
DATASET = "dbpedia_gamma"
N_RANDOM = 8
N_GRID = 40


def _full_blocks(ctx):
    blocks, _ = compute_scc_blocks(ctx.meta_full)
    gar_by_id = {g.id: g for g in ctx.gars_full}
    return [{"gars": [gar_by_id[r["id"]] for r in c if r["id"] in gar_by_id],
             "label": f"blk{i}"} for i, c in enumerate(blocks)]


def _run_order(ctx, tasks, order):
    return AFFRunner(ctx, AFFConfig(task_source="prebuilt", prebuilt_tasks=tasks,
        order_strategy="custom", custom_order=list(order), sharing="shared",
        decision_mode="monotone", canonicalize=True, record_p_out_events=True)).run()


def _warmed_order(ctx, tasks):
    """AFF's learned order: probe FIFO with snapshots, measure per-block P_out gain,
    feed into the reward policy."""
    blocks, _ = compute_scc_blocks(ctx.meta_full)
    dag = task_dag_from_meta(blocks)
    infos = build_task_infos(blocks, ctx.cfg.p_out, dag_edges=dag)
    probe = AFFRunner(ctx, AFFConfig(task_source="prebuilt", prebuilt_tasks=tasks,
        order_strategy="fifo", sharing="shared", canonicalize=False,
        track_snapshots=True)).run()
    prev = 0
    for k, (ci, _rs) in enumerate(probe.per_task):
        cur = len(probe.per_task_snapshots[k]["cumulative_p_out"])
        gain = max(cur - prev, 0); prev = cur
        if 0 <= ci < len(infos):
            infos[ci].stats.update(gain=float(gain), noise=0.0, cost=0.0)
    return RewardGuidedPolicy(adaptive=True, lookahead=True).order(infos, dag=dag)


def _rand_topo(ctx, n, seed):
    import random
    rng = random.Random(seed)
    blocks, _ = compute_scc_blocks(ctx.meta_full)
    dag = task_dag_from_meta(blocks)
    return topo_order(list(range(n)), dag, {i: rng.random() for i in range(n)})


def run() -> dict:
    ctx = load_ctx(DATASET)
    gt, _ = ground_truth_and_total(ctx)
    tasks = _full_blocks(ctx)
    n = len(tasks)
    with _quiet():
        warmed = _warmed_order(ctx, tasks)
        fixed = topo_order(list(range(n)), task_dag_from_meta(
            compute_scc_blocks(ctx.meta_full)[0]), {i: i for i in range(n)})
        raw = {"AFF": [_run_order(ctx, tasks, warmed)],
               "fixed": [_run_order(ctx, tasks, fixed)],
               "random": [_run_order(ctx, tasks, _rand_topo(ctx, n, 100 + s))
                          for s in range(N_RANDOM)]}
    max_t = max(r.timing.get("total", 0.0) for runs in raw.values() for r in runs)
    grid = [max_t * i / (N_GRID - 1) for i in range(N_GRID)]
    per = {}
    for label, runs in raw.items():
        cc = [[row["recall"] for row in curves.coverage_vs_time(
               r.p_out_event_log, gt, ctx.pca_by_gar, ctx.functional_p_out, grid)]
              for r in runs]
        cols = list(zip(*cc))
        per[label] = {"mean": [round(sum(c) / len(c), 4) for c in cols]}
    mid = len(grid) // 2
    print(f"[{SLUG}] {DATASET}  recall@mid: AFF={per['AFF']['mean'][mid]:.3f} "
          f"fixed={per['fixed']['mean'][mid]:.3f} random={per['random']['mean'][mid]:.3f}")
    payload = {"dataset": DATASET, "slug": SLUG, "grid": [round(t, 6) for t in grid],
               "per_order": per}
    d = DATA / SLUG; d.mkdir(parents=True, exist_ok=True)
    (d / f"{DATASET}.json").write_text(json.dumps(payload, indent=2))
    return payload


def plot(payload: dict) -> None:
    plt = P.setup_mpl()
    xs = payload["grid"]
    fig, ax = plt.subplots()
    for label, v in payload["per_order"].items():
        P.line(ax, xs, v["mean"], label, lw=1.4, emphasis=(label == "AFF"))
    ax.set_xlabel("runtime (s)"); ax.set_ylabel("recall"); ax.set_ylim(-0.03, 1.05)
    f = FIGS / SLUG; f.mkdir(parents=True, exist_ok=True)
    fig.savefig(f / f"recall_{DATASET}.pdf"); plt.close(fig)
    print(f"    saved figures/{SLUG}/recall_{DATASET}.pdf")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    pl = json.loads((DATA / SLUG / f"{DATASET}.json").read_text()) if a.plot_only else run()
    plot(pl)
