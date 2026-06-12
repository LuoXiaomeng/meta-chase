"""efficiency — runtime vs the FF baseline as task load grows.

    python -m experiments.showcase.efficiency
"""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

from experiments import datasets
from experiments.experiment import load_ctx, _quiet
from experiments import plot as P
from metachase import build_merged_scc_tasks, AFFConfig, AFFRunner
from metachase.rules.group_rules import compute_scc_blocks

datasets.register_paper_datasets()

SLUG = "efficiency"
DATASET = "dbpedia_gamma"
SHOWCASE = Path(__file__).resolve().parent
KS = [25, 50, 100, 150, 200]
SEEDS = [0, 1, 2]            # fixed shuffle seeds -> reproducible 3-trial average


def _time(ctx, tasks, fifo: bool) -> float:
    if not tasks:
        return 0.0
    cfg = dict(task_source="prebuilt", prebuilt_tasks=tasks, sharing="shared",
               decision_mode="monotone", max_rounds=0)
    if fifo:                                  # AFF topo-orders the relevant blocks itself
        cfg["order_strategy"] = "fifo"
    else:
        cfg["order_strategy"] = "custom"; cfg["custom_order"] = list(range(len(tasks)))
    with _quiet():
        t0 = time.perf_counter()
        AFFRunner(ctx, AFFConfig(**cfg)).run()
        return time.perf_counter() - t0


def _fftask_time(ctx, gars, slots: int) -> float:
    """Literally re-run the full rule set fresh, once per slot (the naive baseline)."""
    from metachase import run_ff_task
    with _quiet():
        t0 = time.perf_counter()
        run_ff_task(ctx, slots, gars=gars, record_events=False, mode="canon")
        return time.perf_counter() - t0


def run() -> dict:
    ctx = load_ctx(DATASET)
    nb = len(compute_scc_blocks(ctx.meta_full)[0])
    stream = build_merged_scc_tasks(ctx, nb)
    rows = []
    for k in KS:
        aff_t, ff_t = [], []
        for sd in SEEDS:
            s = list(stream); random.Random(sd).shuffle(s)
            prefix = s[:k]
            aff_t.append(_time(ctx, [t for t in prefix if t["has_relevant"]], fifo=True))
            allg = [g for t in prefix for g in t["gars"]]
            ff_t.append(_time(ctx, [{"gars": allg, "label": "ff"}], fifo=False))
        aff, ff = round(statistics.mean(aff_t), 4), round(statistics.mean(ff_t), 4)
        s0 = list(stream); random.Random(SEEDS[0]).shuffle(s0)
        ff_task_gars = [g for t in s0[:k] for g in t["gars"]]
        fftask = round(_fftask_time(ctx, ff_task_gars, k), 4)
        rows.append({"N": k, "AFF": aff, "FF": ff, "FF_task": fftask})
    print(f"[{SLUG}] {DATASET} (mean of {len(SEEDS)} shuffles)")
    for r in rows:
        print(f"  N={r['N']:>3}  AFF={r['AFF']:.3f}  FF={r['FF']:.3f}  FF_task={r['FF_task']:.1f}")
    payload = {"slug": SLUG, "dataset": DATASET, "seeds": SEEDS, "rows": rows}
    d = SHOWCASE / "data" / SLUG; d.mkdir(parents=True, exist_ok=True)
    (d / f"{DATASET}.json").write_text(json.dumps(payload, indent=2))
    return payload


def plot(payload: dict) -> None:
    plt = P.setup_mpl()
    fig, ax = plt.subplots()
    rows = payload["rows"]; xs = [r["N"] for r in rows]
    for arm in ["AFF", "FF", "FF_task"]:
        st = P.series_style(arm)
        ax.plot(xs, [r[arm] for r in rows], color=st["color"], ls=st["ls"],
                marker=st["marker"], ms=5, lw=(1.8 if arm == "AFF" else 1.3))
    ax.set_yscale("log"); ax.set_xlabel("N"); ax.set_ylabel("runtime (s)")
    f = SHOWCASE / "figures" / SLUG; f.mkdir(parents=True, exist_ok=True)
    fig.savefig(f / f"time_{DATASET}.pdf"); plt.close(fig)
    print(f"    saved figures/{SLUG}/time_{DATASET}.pdf")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    pl = json.loads((SHOWCASE / "data" / SLUG / f"{DATASET}.json").read_text()) if a.plot_only else run()
    plot(pl)
