"""experiment — the shared run loop: for each arm run `pipeline.run`, record metrics per
(arm, sweep-point) to JSON, then plot. A driver only declares arms / sweep / build / metrics.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import pipeline, plot as P
from .metrics import Metric

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"

_CTX_CACHE: Dict[str, object] = {}


def load_ctx(dataset: str):
    if dataset not in _CTX_CACHE:
        from metachase import Context
        with _quiet():
            _CTX_CACHE[dataset] = Context(dataset)
    return _CTX_CACHE[dataset]


def ground_truth(ctx) -> set:
    """GT = the focused rule set chased to closure and canonicalised (one max-PCA
    value per functional key) — what every arm is scored against."""
    return ground_truth_and_total(ctx)[0]


def ground_truth_and_total(ctx):
    """(GT, ff_total_chasing_steps). GT = focused-canon closure; ff_total is the FF
    run's total chasing steps (used to size the chasing-step axis). Cached per ctx."""
    if not hasattr(ctx, "_gt_total_cache"):
        from metachase import run_ff
        with _quiet():
            ff = run_ff(ctx, record_events=False, canonicalize=True)
        gt = set(ff.p_out)
        total = sum(rs.chasing_steps for _ci, rs in ff.per_task)
        ctx._gt_total_cache = (gt, total)
    return ctx._gt_total_cache


def _rule_stats_for(ctx):
    """Task-independent per-rule warmup stats for the unified policy (cached)."""
    if not hasattr(ctx, "_rule_stats_cache"):
        from . import scheduling
        with _quiet():
            ctx._rule_stats_cache = scheduling.rule_stats_canonical(ctx)
    return ctx._rule_stats_cache


@contextlib.contextmanager
def _quiet():
    """Silence the engine's per-round progress chatter during a run. Redirects at the
    FILE-DESCRIPTOR level (not just sys.stdout) so tqdm/progress writes that go
    straight to fd 1/2 are caught too."""
    saved = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(devnull, 1); os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        os.close(devnull); os.close(saved[0]); os.close(saved[1])


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Sweep:
    axis: str                       # x-axis key, e.g. "k" / "n_tasks"
    values: List                    # the sweep points


@dataclass
class Experiment:
    slug: str
    datasets: List[str]
    arms: List[str]                                 # display order
    build: Callable                                 # (ctx, state, sweep_value) -> {arm: Method}
    metrics: List[Metric]
    sweep: Optional[Sweep] = None
    setup: Optional[Callable] = None                # (ctx) -> state, run once per dataset
    logy: List[str] = field(default_factory=list)   # metric keys drawn on a log y-axis
    plot_fn: Optional[Callable] = None              # custom plot(slug, ds, payload, plt) override
    derived: Dict[str, str] = field(default_factory=dict)  # derived_arm -> base_arm (= base × sweep value)
    out_root: Optional[Path] = None                 # write data/figures here instead of experiments/

    # -- paths ------------------------------------------------------------
    def _data_path(self, ds: str) -> Path:
        d = (self.out_root or ROOT) / "data" / self.slug; d.mkdir(parents=True, exist_ok=True)
        return d / f"{ds}.json"

    def _fig_path(self, name: str) -> Path:
        d = (self.out_root or ROOT) / "figures" / self.slug; d.mkdir(parents=True, exist_ok=True)
        return d / f"{name}.pdf"

    # -- run --------------------------------------------------------------
    def run(self) -> "Experiment":
        needs_gt = any(m.needs_gt for m in self.metrics)
        for ds in self.datasets:
            ctx = load_ctx(ds)
            state = self.setup(ctx) if self.setup else None
            gt = ground_truth(ctx) if needs_gt else None
            sweep_vals = self.sweep.values if self.sweep else [None]
            axis = self.sweep.axis if self.sweep else "_"
            print(f"\n[{self.slug}] {ds}")
            rows = []
            for v in sweep_vals:
                methods = self.build(ctx, state, v)
                row = {} if v is None else {axis: v}
                for arm in self.arms:
                    m = methods.get(arm)
                    if m is None:
                        continue
                    with _quiet():
                        tr = pipeline.run(ctx, m)
                    row[arm] = {met.key: met(tr, gt) for met in self.metrics}
                # derived arms = a measured arm scaled by the sweep value (e.g.
                # FF_task = FF × N, the N-fresh-reruns baseline — no need to run it).
                for da, base in self.derived.items():
                    if base in row and v is not None:
                        row[da] = {k: vv * v for k, vv in row[base].items()}
                rows.append(row)
                self._log_row(axis, v, row)
            payload = {"dataset": ds, "slug": self.slug, "axis": axis,
                       "values": [x for x in sweep_vals if x is not None],
                       "arms": self.arms, "rows": rows,
                       "metrics": [m.key for m in self.metrics]}
            self._data_path(ds).write_text(json.dumps(payload, indent=2, default=str))
        return self

    def _log_row(self, axis, v, row):
        head = "" if v is None else f"{axis}={v:<5}"
        cells = []
        for arm in self.arms:
            if arm in row:
                cells.append(arm + " " + " ".join(f"{k}={val}" for k, val in row[arm].items()))
        print(f"    {head} | " + " | ".join(cells))

    # -- plot -------------------------------------------------------------
    def plot(self) -> "Experiment":
        for ds in self.datasets:
            p = self._data_path(ds)
            if not p.exists():
                print(f"    [{self.slug}] no data for {ds}; run() first")
                continue
            payload = json.loads(p.read_text())
            if self.plot_fn:
                plt = P.setup_mpl()
                self.plot_fn(self.slug, ds, payload, plt, self)
            else:
                self._default_line_plot(ds, payload)
        return self

    def _default_line_plot(self, ds: str, payload: dict) -> None:
        """One line figure per metric: x = sweep axis, one series per arm."""
        plt = P.setup_mpl()
        axis = payload["axis"]
        rows = payload["rows"]
        if not rows or axis == "_":
            return
        for met in self.metrics:
            fig, ax = plt.subplots()
            for arm in self.arms:
                xs = [r[axis] for r in rows if arm in r]
                ys = [r[arm][met.key] for r in rows if arm in r]
                if not xs:
                    continue
                P.line(ax, xs, ys, arm, lw=1.4, emphasis=(arm == "AFF"))
            ax.set_xlabel(axis); ax.set_ylabel(met.label)
            if met.key in self.logy:
                ax.set_yscale("log")
            path = self._fig_path(f"{met.key}_{ds}")
            fig.savefig(path); plt.close(fig)
            print(f"    saved {path.relative_to(self.out_root or ROOT)}")


@dataclass
class CurveExperiment:
    """For Exp-1-style experiments: no sweep axis — each arm runs ONCE (with the event
    log on) and a post-hoc `curve_fn` reconstructs a series from its event log (e.g.
    precision/recall vs chasing steps). One figure per `y`, one line per arm."""
    slug: str
    datasets: List[str]
    arms: List[str]
    build: Callable                          # (ctx) -> {arm: Method}
    curve_fn: Callable                       # (trace, gt, ctx, axis) -> list[dict]
    ys: List[str]                            # row keys to plot (e.g. precision/recall)
    x: str = "step"
    n_points: int = 20
    xmax: Optional[int] = None               # cap x-axis when plotting
    ylim: Optional[tuple] = None
    out_root: Optional[Path] = None          # write data/figures here instead of experiments/

    def _data_path(self, ds: str) -> Path:
        d = (self.out_root or ROOT) / "data" / self.slug; d.mkdir(parents=True, exist_ok=True)
        return d / f"{ds}.json"

    def _fig_path(self, name: str) -> Path:
        d = (self.out_root or ROOT) / "figures" / self.slug; d.mkdir(parents=True, exist_ok=True)
        return d / f"{name}.pdf"

    def run(self) -> "CurveExperiment":
        from .curves import step_targets
        for ds in self.datasets:
            ctx = load_ctx(ds)
            gt, ff_total = ground_truth_and_total(ctx)
            interval = max(1, ff_total // self.n_points)
            methods = self.build(ctx)
            print(f"\n[{self.slug}] {ds}  |GT|={len(gt)}  ff_total={ff_total}"
                  f"  interval={interval}")
            curves, totals = {}, {}
            for arm in self.arms:
                m = methods.get(arm)
                if m is None:
                    continue
                with _quiet():
                    tr = pipeline.run(ctx, m)
                axis = step_targets(tr.total_steps, interval)
                curves[arm] = self.curve_fn(tr, gt, ctx, axis)
                totals[arm] = tr.total_steps
                last = curves[arm][-1] if curves[arm] else {}
                print(f"    {arm:5s} steps={tr.total_steps:>8}  " +
                      "  ".join(f"{y}={last.get(y, 0):.3f}" for y in self.ys))
            payload = {"dataset": ds, "slug": self.slug, "x": self.x, "ys": self.ys,
                       "arms": self.arms, "gt_size": len(gt), "interval": interval,
                       "totals": totals, "curves": curves}
            self._data_path(ds).write_text(json.dumps(payload, indent=2, default=str))
        return self

    def plot(self) -> "CurveExperiment":
        for ds in self.datasets:
            p = self._data_path(ds)
            if not p.exists():
                print(f"    [{self.slug}] no data for {ds}; run() first")
                continue
            payload = json.loads(p.read_text())
            plt = P.setup_mpl()
            for y in self.ys:
                fig, ax = plt.subplots()
                for arm in self.arms:
                    curve = payload["curves"].get(arm) or []
                    pts = [(c[self.x], c[y]) for c in curve
                           if self.xmax is None or c[self.x] <= self.xmax]
                    if not pts:
                        continue
                    P.line(ax, [q[0] for q in pts], [q[1] for q in pts], arm,
                           lw=1.6, emphasis=(arm == "AFF"))
                ax.set_xlabel(self.x); ax.set_ylabel(y)
                if self.xmax is not None:
                    ax.set_xlim(0, self.xmax * 1.04)
                if self.ylim is not None:
                    ax.set_ylim(*self.ylim)
                path = self._fig_path(f"{y}_{ds}")
                fig.savefig(path); plt.close(fig)
                print(f"    saved {path.relative_to(self.out_root or ROOT)}")
        return self
