"""Vanilla Fishing Fort coordinator — chase / fixpoint semantics."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple  # noqa: F401

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from . import primitive as chase
from ..graph.derived import DerivedFacts, GraphView
from ..graph.gar import GAR, ConclusionType
from ..graph.graph import Graph
from ..match.matcher import Matcher


# Per-round statistics

@dataclass
class RoundStats:
    """Statistics collected for one chase round."""
    round_num:     int
    elapsed:       float          # wall-clock seconds for this round
    new_facts:     int            # new edges/attrs derived
    n_active:      int            # rules that ran
    n_skipped:     int            # rules skipped by semi-naive
    overlay_edges: int            # overlay size at round start
    overlay_attrs: int
    delta_labels:  int            # distinct edge label types in prev delta
    delta_attrs:   int            # distinct attr names in prev delta
    top_rules:     "List[Tuple[str, int]]"  # (gar_id, new_facts) top-10
    new_p_out:     int = 0        # NEW P_out facts this round (if p_out_ids set)
    cum_p_out:     int = 0        # cumulative P_out facts after this round
    gen_events:    int = 0        # ALL fact-generation events this round (incl. dups)


@dataclass
class RunStats:
    """Aggregate statistics for one full coordinator.run() invocation."""
    task_id:           "Optional[str]" = None
    total_rounds:      int   = 0
    total_elapsed:     float = 0.0
    chasing_steps:     int   = 0
    rule_invocations:  int   = 0
    gen_events:        int   = 0
    new_facts:         int   = 0
    converged:         bool  = False


# Result record

@dataclass
class FiringResult:
    """One concrete firing of a GAR on a match."""
    gar_id:     str
    round:      int                     # which chase round produced this
    conclusion: ConclusionType
    match:      Dict[str, int]          # var → node_id
    output:     Any                     # payload (depends on conclusion type)
    label:      str = ""
    task_id:    "Optional[str]" = None  # which Meta-Chase task produced this

    def pretty(self, graph) -> str:
        """Human-readable one-liner."""
        def fmt(nid: int) -> str:
            return f"{nid}({graph.type_name(nid)})"

        vars_str = ", ".join(f"{v}={fmt(nid)}" for v, nid in self.match.items())

        if self.conclusion == ConclusionType.ENRICH:
            src_nid, elid, dst_nid = self.output
            out_str = f"{fmt(src_nid)} -[{graph.edge_name(elid)}]→ {fmt(dst_nid)}"
        elif self.conclusion == ConclusionType.DEDUCE:
            var, attr, val = self.output
            out_str = f"{var}.{attr} = {val}"
        else:
            out_str = f"flag {self.output}"

        tag = f"[{self.label}]" if self.label else ""
        return (f"r{self.round} GAR({self.gar_id}){tag}"
                f"  [{vars_str}]  →  {out_str}")


# Coordinator

class VanillaCoordinator:
    """Baseline Fishing Fort execution engine with chase / fixpoint semantics."""

    def __init__(
        self,
        graph:       Graph,
        gars:        List[GAR],
        max_rounds:  int  = 0,
        max_matches: int  = 0,
        max_derived: int  = 0,
        verbose:     bool = True,
        matcher:     "Optional[Matcher]" = None,
        p_out_ids:   "Optional[Set[int]]" = None,
        functional_p_out: "Optional[Set[int]]" = None,
        early_commit: bool = False,
        record_p_out_events: bool = False,
    ) -> None:
        self.graph       = graph
        self.gars        = gars
        self.max_rounds  = max_rounds
        self.max_matches = max_matches
        self.max_derived = max_derived
        self.verbose     = verbose
        self._matcher    = matcher if matcher is not None else Matcher(graph, max_matches=max_matches)
        self.round_stats: List[RoundStats] = []
        self.run_stats:   RunStats         = RunStats()
        # Optional: edge-label IDs counted as P_out (decision outputs).
        self.p_out_ids   = p_out_ids
        # Optional: edge-label IDs of FUNCTIONAL (single-valued) P_out predicates.
        self.functional_p_out = functional_p_out
        # Early-commit (first-writer-wins) on functional keys. OFF by default ->
        self.early_commit = early_commit
        # Provenance for canonicalization: P_out triple -> set of producing gar ids.
        self.p_out_provenance: Dict[Any, set] = {}
        # Optional ordered, timestamped log of EVERY P_out producing-firing (new
        self.record_p_out_events = record_p_out_events
        self.p_out_events: "List[tuple]" = []

    # Main entry point

    def run(
        self,
        existing_derived:     "Optional[DerivedFacts]" = None,
        existing_fired_pairs: "Optional[Set]"          = None,
        task_id:              "Optional[str]"          = None,
        on_round_end:         "Optional[Any]"          = None,
        on_gar_end:           "Optional[Any]"          = None,
        seed_delta_edges:     "Optional[List]"         = None,
        seed_delta_attrs:     "Optional[Set]"          = None,
        incremental:          bool                     = False,
    ) -> List[FiringResult]:
        """Execute all GARs to fixpoint and return all firing results."""
        derived     = existing_derived if existing_derived is not None else DerivedFacts()
        all_results: List[FiringResult] = []
        self.p_out_provenance = {}      # fresh per run; caller merges across coords
        self.p_out_events = []          # fresh per run; caller concatenates across coords
        self._cum_steps = 0             # running TOTAL new-fact count (all predicates)
        # New edges/attrs THIS run derives, accumulated across its rounds, so a caller
        self.produced_edges: List[Tuple[int, int, int]] = []
        self.produced_attrs: Set[str] = set()
        self._incremental_run = bool(incremental)
        # Discard any edges/attrs that accumulated before this run started — they don't
        derived.take_delta_edges()
        derived.take_delta_attrs()

        # fired_pairs passed in by the caller is kept for cross-task dedup;
        external_fired = existing_fired_pairs if existing_fired_pairs is not None else None

        # Python semi-naive (overlay-anchored) matching: only sound for a MONOTONE
        self._seminaive_ok = (
            os.environ.get("FF_PY_SEMINAIVE", "0") == "1"
            and self._matcher.backend == "python"
            and not self.early_commit
            and on_round_end is None
            and on_gar_end is None
            and external_fired is not None
        )

        wall_start = time.perf_counter()
        round_num  = 0
        converged  = False
        cum_p_out  = 0   # cumulative P_out facts (when p_out_ids set)
        # Round-to-round delta that drives the rule-level skip (can_skip). Normally
        seed_edges = list(seed_delta_edges) if (incremental and seed_delta_edges) else []
        delta_labels: Set[int] = {l for (_s, _d, l) in seed_edges}
        delta_attr_names: Set[str] = (set(seed_delta_attrs)
                                      if (incremental and seed_delta_attrs) else set())
        # Delta edge arrays for semi-naive evaluation are unused by the pure-Python
        delta_arrays: "Optional[tuple]" = None  # (src_arr, dst_arr, lbl_arr)

        while True:
            round_num += 1
            round_start   = time.perf_counter()
            overlay_edges = derived.num_edges
            overlay_attrs = derived.num_attrs
            view      = GraphView(self.graph, derived)
            new_facts = 0
            round_gen = 0

            # Per-round fired-pairs set — released at end of round to bound memory.
            fired_pairs: Set = set() if external_fired is None else external_fired

            n_gars = len(self.gars)
            overlay_info = (f"  overlay: {derived.num_edges} edges, "
                            f"{derived.num_attrs} attrs" if round_num > 1 else "")

            # Compute how many GARs will be skipped this round.
            if (self._incremental_run
                    or (round_num > 1 and (delta_labels or delta_attr_names))):
                def _gar_is_active(g: GAR) -> bool:
                    # Wildcard edge (label=None) matches any derived edge.
                    if any(e.label is None or e.label in delta_labels
                           for e in g.pattern.edges):
                        return True
                    for pred in g.predicates:
                        if pred.has_unknown_attr_deps:
                            return True
                        if pred.referenced_attrs & delta_attr_names:
                            return True
                    return False
                n_active  = sum(1 for g in self.gars if _gar_is_active(g))
                n_skipped = n_gars - n_active
                can_skip  = True
            else:
                n_active, n_skipped = n_gars, 0
                can_skip = False

            # Round info goes into the tqdm description so we get ONE
            if round_num == 1:
                tqdm_desc = f"R 1 │ovly     0 │act {n_gars:>3}/{n_gars}"
            else:
                tqdm_desc = (
                    f"R{round_num:2d} │ovly {overlay_edges:>5}"
                    f" │act {n_active:>3}/{n_gars}"
                    + (f" skip {n_skipped}" if n_skipped else "")
                )
            gar_iter = enumerate(self.gars)
            if _TQDM_AVAILABLE:
                gar_iter = enumerate(_tqdm(
                    self.gars,
                    desc=tqdm_desc,
                    unit="rule",
                    leave=False,
                    dynamic_ncols=True,
                    file=sys.stderr,
                ))
            else:
                # No tqdm available — fall back to a single banner line per
                print(f"  [{tqdm_desc}]", flush=True)
                if self.verbose:
                    print(f"    Running {n_active} active rules ...", flush=True)

            round_results: List[FiringResult] = []
            for gar_idx, gar in gar_iter:
                # Delta skip: only when can_skip (see above).
                if can_skip and not _gar_is_active(gar):
                    continue

                gar_new, gar_gen, gar_fired = self._run_one_gar(
                    gar, gar_idx, n_gars, round_num, view, derived, fired_pairs,
                    delta_arrays=delta_arrays,
                    task_id=task_id,
                )
                new_facts += gar_new
                round_gen += gar_gen
                round_results.extend(gar_fired)

                # Finest-grain opt-in hook (default None -> no effect): fires AFTER
                if on_gar_end is not None:
                    on_gar_end(self, gar, derived, round_num)

            all_results.extend(round_results)

            # Collect delta labels for next round from this round's new ENRICH facts.
            from ..graph.gar import ConclusionType
            delta_labels = {
                r.output[1] for r in round_results
                if r.conclusion == ConclusionType.ENRICH
            }
            # Collect attribute names newly set this round (DEDUCE/CLEAN).
            delta_attr_names = derived.take_delta_attrs()

            # take_delta_edges() returns edges added during *this* round and resets the buffer.
            round_delta = derived.take_delta_edges()
            # Accumulate this run's own output so a caller can seed a follow-up
            self.produced_edges.extend(round_delta)
            self.produced_attrs |= delta_attr_names
            # Pure-Python backend does a full re-match each round (dedup via
            delta_arrays = None

            # Release per-round fired-pairs (unless shared with caller).
            if external_fired is None:
                fired_pairs = set()

            # Collect per-round stats (always, not just when verbose).
            by_gar: Dict[str, int] = {}
            for r in round_results:
                by_gar[r.gar_id] = by_gar.get(r.gar_id, 0) + 1
            top_rules = sorted(by_gar.items(), key=lambda x: -x[1])[:10]

            # P_out tracking: count NEW ENRICH facts whose label is in p_out_ids.
            new_p_out = 0
            if self.p_out_ids:
                for r in round_results:
                    if (r.conclusion == ConclusionType.ENRICH
                            and r.output[1] in self.p_out_ids):
                        new_p_out += 1
                cum_p_out += new_p_out

            self.round_stats.append(RoundStats(
                round_num=round_num,
                elapsed=time.perf_counter() - round_start,
                new_facts=new_facts,
                n_active=n_active,
                n_skipped=n_skipped,
                overlay_edges=overlay_edges,
                overlay_attrs=overlay_attrs,
                delta_labels=len(delta_labels),
                delta_attrs=len(delta_attr_names),
                top_rules=top_rules,
                new_p_out=new_p_out,
                cum_p_out=cum_p_out,
                gen_events=round_gen,
            ))

            # Optional round-end hook (opt-in; default None -> no effect). Lets a
            if on_round_end is not None:
                on_round_end(self, derived, all_results, round_num)

            if self.verbose:
                p_out_str = (f"  P_out: +{new_p_out} (cum {cum_p_out})"
                             if self.p_out_ids else "")
                print(f"  → {new_facts} new facts this round"
                      + (f"  |delta_labels|={len(delta_labels)}" if delta_labels else "")
                      + p_out_str,
                      flush=True)
                if top_rules:
                    print("  Top rules this round:", flush=True)
                    for gid, cnt in top_rules:
                        print(f"    {gid:30s}  {cnt:,}", flush=True)
            else:
                # Concise always-on per-round heartbeat (so long single-task
                _rt = time.perf_counter() - round_start
                _po = f"  P_out cum {cum_p_out:,}" if self.p_out_ids else ""
                print(f"    R{round_num:<2d} +{new_facts:>8,} facts  "
                      f"cum {derived.num_edges:>9,} edges  ({_rt:5.1f}s){_po}",
                      flush=True)

            # Fixpoint: no new facts produced
            if new_facts == 0:
                if self.verbose:
                    print(f"  fixpoint reached after {round_num} round(s)")
                converged = True
                break

            if self.max_rounds > 0 and round_num >= self.max_rounds:
                if self.verbose:
                    print(f"  max_rounds={self.max_rounds} reached; stopping")
                converged = False
                break

            total_derived = derived.num_edges + derived.num_attrs
            if self.max_derived > 0 and total_derived >= self.max_derived:
                if self.verbose:
                    print(f"  max_derived={self.max_derived} reached "
                          f"({total_derived} facts); stopping")
                converged = False
                break

        elapsed = time.perf_counter() - wall_start

        # Aggregate RunStats: chasing_steps = sum of new_facts; invocations = active rules.
        self.run_stats = RunStats(
            task_id=task_id,
            total_rounds=round_num,
            total_elapsed=elapsed,
            chasing_steps=sum(rs.new_facts for rs in self.round_stats),
            rule_invocations=sum(rs.n_active for rs in self.round_stats),
            gen_events=sum(rs.gen_events for rs in self.round_stats),
            new_facts=len(all_results),
            converged=converged,
        )

        if self.verbose:
            print(
                f"\n[Coordinator] finished  "
                f"{len(self.gars)} GARs  ·  "
                f"{round_num} rounds  ·  "
                f"{len(all_results)} total firings  "
                f"({elapsed:.2f}s)  "
                f"steps={self.run_stats.chasing_steps:,}  "
                f"invocations={self.run_stats.rule_invocations:,}  "
                f"converged={converged}"
            )
        return all_results

    # Per-GAR execution within one round

    def _gar_edge_only(self, gar: GAR) -> bool:
        """True if `gar`'s firing depends ONLY on its edge pattern — no predicate reads a (possibly derived) attribute and none has unknown attr deps."""
        cache = getattr(self, "_edge_only_cache", None)
        if cache is None:
            cache = self._edge_only_cache = {}
        val = cache.get(gar.id)
        if val is None:
            val = all((not p.has_unknown_attr_deps) and (not p.referenced_attrs)
                      for p in gar.predicates)
            cache[gar.id] = val
        return val

    def _run_one_gar(
        self,
        gar:          GAR,
        gar_idx:      int,
        n_gars:       int,
        round_num:    int,
        view:         GraphView,
        derived:      DerivedFacts,
        fired_pairs:  Set[Tuple[str, FrozenSet]],
        delta_arrays: "Optional[tuple]" = None,
        task_id:      "Optional[str]"   = None,
    ) -> Tuple[int, List[FiringResult]]:
        """Run one GAR against *view* for this round."""
        t0        = time.perf_counter()
        results:  List[FiringResult] = []
        new_facts = 0
        gen       = 0   # every fact-generation event (new OR re-derived) = paper rule-invocations

        # Match-source selection (round 2+ only differs from a full match):
        if (self._seminaive_ok and (round_num >= 2 or self._incremental_run)
                and self._gar_edge_only(gar)):
            match_iter = self._matcher.match_overlay(gar.pattern, view)
        elif delta_arrays is not None:
            match_iter = self._matcher.match_delta(
                gar.pattern, view, *delta_arrays
            )
        else:
            match_iter = self._matcher.match(gar.pattern, graph=view)

        # Build a stable, sorted variable order ONCE per rule. Per-match we then
        ordered_vars = sorted(gar.pattern.nodes.keys())
        gar_id = gar.id  # local for tight loop

        for match in match_iter:
            pair_key = (gar_id, tuple(match[v] for v in ordered_vars))
            if pair_key in fired_pairs:
                continue

            if not gar.check_predicates(match, view):
                continue

            is_new, output = self._materialize(gar, match, derived)
            fired_pairs.add(pair_key)
            gen += 1   # a fact was generated (counts even if it duplicates an existing one)

            # Record provenance for EVERY P_out producer (new or not) so canonical
            if self.p_out_ids and isinstance(output, tuple) and len(output) == 3 \
                    and output[1] in self.p_out_ids:
                self.p_out_provenance.setdefault(output, set()).add(gar_id)
                # Ordered, timestamped event log (B1) — recorded BEFORE the is_new
                if self.record_p_out_events:
                    self.p_out_events.append(
                        (output[0], output[1], output[2], gar_id, is_new,
                         time.perf_counter(),
                         self._cum_steps + (1 if is_new else 0)))

            if not is_new:
                continue

            new_facts += 1
            self._cum_steps += 1
            results.append(FiringResult(
                gar_id=gar_id,
                round=round_num,
                conclusion=gar.conclusion.type,
                match=match,
                output=output,
                label=gar.conclusion.label,
                task_id=task_id,
            ))

        elapsed = time.perf_counter() - t0
        if self.verbose and not _TQDM_AVAILABLE:
            # Plain progress line: rule index / total, id, new facts, time
            print(
                f"  [{gar_idx+1:3d}/{n_gars}] {gar.id:20s}  "
                f"new={new_facts:6d}  ({elapsed:.2f}s)",
                flush=True,
            )
        return new_facts, gen, results

    # Conclusion materialization

    def _materialize(
        self,
        gar:     GAR,
        match:   Dict[str, int],
        derived: DerivedFacts,
    ) -> Tuple[bool, Any]:
        """Apply the conclusion to *derived*."""
        return chase.materialize(gar, match, self.graph, derived,
                                 functional_keys=self.functional_p_out,
                                 early_commit=self.early_commit)

    # Report

    def report(
        self,
        results: List[FiringResult],
        limit:   int = 20,
    ) -> str:
        lines = [
            "=== Vanilla Fishing Fort Report (chase / fixpoint) ===",
            f"Total firings : {len(results)}",
            "",
        ]
        by_gar: Dict[str, List[FiringResult]] = {}
        for r in results:
            by_gar.setdefault(r.gar_id, []).append(r)

        for gid, rs in by_gar.items():
            # Group by round
            by_round: Dict[int, int] = {}
            for r in rs:
                by_round[r.round] = by_round.get(r.round, 0) + 1
            round_summary = ", ".join(
                f"r{k}:{v}" for k, v in sorted(by_round.items())
            )
            lines.append(f"--- {gid}  ({len(rs)} firings  [{round_summary}]) ---")
            for r in rs[:limit]:
                lines.append("  " + r.pretty(self.graph))
            if len(rs) > limit:
                lines.append(f"  ... and {len(rs) - limit} more")
            lines.append("")
        return "\n".join(lines)
