"""recall — recall of the published P_out vs the focused-canon GT over chasing steps, for
AFF, FF and early-commit.

    python -m experiments.showcase.recall
"""

from __future__ import annotations

from pathlib import Path

from experiments import datasets, curves
from experiments.spec import Method, Partition, Schedule, Decision
from experiments.experiment import CurveExperiment

datasets.register_paper_datasets()

SHOWCASE = Path(__file__).resolve().parent
CHASING_XMAX = 40000


def build(ctx):
    return {
        "AFF": Method("AFF", Partition.scc_blocks(), Schedule.natural(),
                      Decision.canon(), runner="aff_unified"),
        "FF":  Method("FF", Partition.all_rules(focused=False), Schedule.natural(),
                      Decision.canon(), shuffle_seed=0),
        "EC":  Method("EC", Partition.all_rules(focused=False), Schedule.natural(),
                      Decision.ec(), shuffle_seed=0),
    }


def _accuracy(tr, gt, ctx, axis):
    return curves.accuracy_curve(tr.event_log, gt, ctx.pca_by_gar,
                                 ctx.functional_p_out, axis)


EXPERIMENT = CurveExperiment(
    slug="recall",
    datasets=["dbpedia_gamma"],
    arms=["AFF", "FF", "EC"],
    build=build,
    curve_fn=_accuracy,
    ys=["recall"],
    x="step",
    n_points=20,
    xmax=CHASING_XMAX,
    ylim=(-0.03, 1.05),
    out_root=SHOWCASE,
)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    (EXPERIMENT if a.plot_only else EXPERIMENT.run()).plot()
