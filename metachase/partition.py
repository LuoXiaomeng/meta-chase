"""partition — task-construction strategies for the chase."""

from __future__ import annotations

import random
from typing import List


def build_random_partition_tasks(ctx, n_tasks: int, seed: int = 42,
                                 scope: str = "focused") -> List[dict]:
    """Randomly assign Σ's rules into ``n_tasks`` round-robin buckets after a seeded shuffle."""
    if scope == "focused":
        gars = list(ctx.gars_focused)
    elif scope == "full":
        gars = list(ctx.gars_full)
    else:
        raise ValueError(f"scope must be 'focused' or 'full', got {scope!r}")
    rng = random.Random(seed)
    rng.shuffle(gars)
    n = max(1, min(n_tasks, len(gars)))
    buckets: List[List] = [[] for _ in range(n)]
    for i, g in enumerate(gars):
        buckets[i % n].append(g)
    return [{"gars": b, "label": f"part{i}", "n_gars": len(b)}
            for i, b in enumerate(buckets) if b]


def build_merged_scc_tasks(ctx, n_tasks: int) -> List[dict]:
    """Merge the full-set SCC blocks into exactly ``n_tasks`` contiguous-topological buckets (the "vary #tasks" partition)."""
    from .rules.group_rules import compute_scc_blocks, relevant_rules
    from .policy import topo_order

    blocks, dag = compute_scc_blocks(ctx.meta_full)
    rel_ids = {r["id"] for r in relevant_rules(ctx.meta_full, set(ctx.cfg.p_out))}
    gar_by_id = {g.id: g for g in ctx.gars_full}
    nb = len(blocks)
    order = topo_order(list(range(nb)), dag, {i: i for i in range(nb)})

    n = max(1, min(n_tasks, nb))
    base, extra = divmod(nb, n)
    tasks, i = [], 0
    for b in range(n):
        size = base + (1 if b < extra else 0)
        bucket = order[i:i + size]; i += size
        rules = [r for bi in bucket for r in blocks[bi]]
        gars = [gar_by_id[r["id"]] for r in rules if r["id"] in gar_by_id]
        tasks.append({
            "gars": gars, "label": f"m{b}", "n_gars": len(gars),
            "n_blocks": len(bucket),
            "has_relevant": any(r["id"] in rel_ids for r in rules),
        })
    return tasks
