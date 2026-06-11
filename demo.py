#!/usr/bin/env python3
"""End-to-end demo of the Meta-Chase (AFF) engine on a tiny bundled dataset.

Run:

    python demo.py

It loads the self-contained `demo` graph + rule set (sample_data/demo),
builds the SCC-block task partition, and runs a task-aware chase via `AFFRunner`.
It prints the derived P_out (decision) facts under two settings:

  1. monotone           -- every candidate value is derived (multi-valued output);
  2. monotone + canon   -- competing values on a FUNCTIONAL P_out key are resolved
                           to one canonical value by max-PCA (paper Sec V.B).

No external data directory is required; everything runs on the pure-Python VF2
matcher (no native/CUDA library needed).
"""
from metachase import AFFConfig, AFFRunner, Context


def name_of(graph, nid):
    """Best-effort human label for a node id (falls back to the id)."""
    inv = getattr(graph, "_demo_names", None)
    if inv is None:
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parent / "sample_data" \
            / "demo" / "processed" / "node_id_dict.json"
        inv = json.loads(p.read_text()) if p.exists() else {}
        graph._demo_names = inv
    return inv.get(str(nid), str(nid))


def show_p_out(ctx, triples, title):
    print(f"\n  {title}  ({len(triples)} facts)")
    g = ctx.graph
    for (s, l, o) in sorted(triples):
        print(f"    {name_of(g, s):10s} -[{g.edge_name(l)}]-> {name_of(g, o)}")


def main():
    print("=" * 60)
    print("Meta-Chase (AFF) engine demo")
    print("=" * 60)

    # 1. Load the bundled demo graph + rules + SCC-block tasks.
    ctx = Context("demo")

    # 2. Run the task-aware chase (SCC blocks, natural order, shared overlay,
    #    monotone decisions) -- the default AFF configuration.
    res_mono = AFFRunner(ctx, AFFConfig(max_rounds=0)).run()

    # 3. Same chase, but resolve functional P_out conflicts by max-PCA (Sec V.B).
    res_canon = AFFRunner(ctx, AFFConfig(max_rounds=0, canonicalize=True)).run()

    print("\n" + "-" * 60)
    print(f"tasks (SCC blocks): {ctx.n_tasks}   "
          f"GAR firings: {len(res_mono.all_results)}   "
          f"chase time: {res_mono.timing['total']*1e3:.1f} ms")
    show_p_out(ctx, res_mono.p_out, "P_out  [monotone, multi-valued]")
    show_p_out(ctx, res_canon.p_out, "P_out  [canonicalized, max-PCA]")

    print("\n" + "=" * 60)
    print(f"RESULT: engine derived {len(res_mono.p_out)} P_out facts "
          f"({len(res_canon.p_out)} after canonicalization).")
    print("=" * 60)
    return res_mono


if __name__ == "__main__":
    main()
