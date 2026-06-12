"""plot — shared paper figure style: one (linestyle, marker, color) per series at the paper's
subfigure size. Drivers call `setup_mpl()` and look colours up via `series_style(name)`.
"""

from __future__ import annotations

# Paper sub-figure slot is 3.3cm × 2.2cm (aspect 1.5). Author at this exact aspect so
# \includegraphics[width=3.3cm,height=2.2cm] places it with no distortion.
PAPER_ASPECT = 3.3 / 2.2                       # = 1.5
PAPER_FIGSIZE = (3.3, 3.3 / PAPER_ASPECT)      # (3.3, 2.2) inches — exact slot ratio
CHASING_XMAX = 40000                           # chasing-steps x cap

SERIES_STYLE = {
    # ---------- scheduling policies (Exp-1 .. 3) ----------
    "AFF":          dict(ls="-",               marker="o", color="#0072B2"),
    "AFF no-reuse": dict(ls=(0, (4, 1, 1, 1)), marker="P", color="#E69F00"),
    "FF":           dict(ls="--",              marker="s", color="#000000"),
    "FF_task":      dict(ls=":",               marker="^", color="#D55E00"),
    "EC":           dict(ls="-.",              marker="D", color="#CC79A7"),
    "random":       dict(ls=(0, (5, 2)),       marker="v", color="#999999"),
    "FIFO":         dict(ls=(0, (1, 1)),       marker="x", color="#56B4E9"),
    "greedy":       dict(ls=(0, (3, 1, 1, 1)), marker="*", color="#009E73"),
    # ---------- confluence-enforcement family (Exp-4.3/4.4) ----------
    "canon":        dict(ls=(0, (6, 3)),       marker="H", color="#44AA99"),
    "two-phase":    dict(ls=(0, (2, 1, 2, 4)), marker="p", color="#332288"),
    "hybrid":       dict(ls=(0, (1, 1, 4, 1)), marker=">", color="#8C510A"),
}
SERIES_ALIAS = {
    "learned": "AFF", "AFF policy": "AFF", "AFF policy (mean)": "AFF",
    "AFF (mean)": "AFF", "random (mean)": "random", "fixed": "FIFO",
    "AFF no reuse": "AFF no-reuse", "no-reuse": "AFF no-reuse",
    "tp-canon": "hybrid",
}

# Per-dataset short tag + colour (Okabe-Ito; consistent across figures).
DS_TAG = {
    "dbpedia_smoke_07_scalable_plus_conf": ("DBpedia", "#0072B2"),
    "dbpedia_smoke_07_scalable_plus":      ("DBpedia", "#0072B2"),
    "dbpedia_smoke_v1":                    ("DBpedia", "#0072B2"),
    "yago310_smoke_acyc2":                 ("YAGO",    "#009E73"),
    "imdb_smoke_dag_rich":                 ("IMDB",    "#E69F00"),
}


def series_style(name: str) -> dict:
    """(linestyle, marker, color) for a series name (alias-resolved). Unknown names
    fall back to neutral grey so nothing crashes."""
    canon = SERIES_ALIAS.get(name, name)
    return dict(SERIES_STYLE.get(canon, dict(ls="-", marker="o", color="#444444")))


def tag(ds: str) -> str:
    return DS_TAG.get(ds, (ds, "#444444"))[0]


def color(ds: str) -> str:
    return DS_TAG.get(ds, (ds, "#444444"))[1]


def line(ax, xs, ys, name, lw=1.4, ms=5, emphasis=False):
    """Draw one paper-style series line: the series' (ls, marker, color), markers at
    every ~len/6 points (ms=5), emphasised (AFF) lines at lw=1.8. No legend — the
    paper carries a single shared legend strip."""
    st = series_style(name)
    me = max(1, len(xs) // 6)
    ax.plot(xs, ys, color=st["color"], ls=st["ls"], marker=st["marker"],
            markevery=me, ms=ms, lw=(1.8 if emphasis else lw))


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": PAPER_FIGSIZE,
        "figure.autolayout": True,
        "savefig.bbox": None,
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.grid": True, "grid.alpha": 0.3,
        "legend.fontsize": 5.5, "legend.frameon": False,
        "lines.linewidth": 1.2, "lines.markersize": 3,
    })
    return plt
