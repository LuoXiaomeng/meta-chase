"""AFFRunner — single entry that dispatches every AFF configuration."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .. import confluence
from ..chase.coordinator import FiringResult, RunStats, VanillaCoordinator
from ..graph.derived import DerivedFacts
from .config import AFFConfig


# Triple = (subject_id, label_id, object_id) — the P_out fact shape.
Triple = Tuple[int, int, int]


@dataclass
class AFFResult:
    """Everything an AFF run produces."""

    p_out: Set[Triple]                              # the raw P_out triple set
    canonical_decisions: Optional[Dict] = None      # {(subj, pred) -> obj} when canonicalize=True
    all_results: List[FiringResult] = field(default_factory=list)
    per_task: List[Tuple[Any, RunStats]] = field(default_factory=list)   # (task_id_payload, RunStats)
    provenance: Dict[Triple, Set[str]] = field(default_factory=dict)     # P_out triple -> producing gar_ids
    timing: Dict[str, float] = field(default_factory=dict)               # 'total', 'per_task': [...]
    config: Optional[AFFConfig] = None              # echo back for traceability
    # When config.track_snapshots=True, one entry per task with the cumulative
    per_task_snapshots: List[Dict] = field(default_factory=list)
    # When config.record_p_out_events=True, the ordered, timestamped P_out event
    p_out_event_log: List = field(default_factory=list)


class AFFRunner:
    """Single facade for every AFF configuration."""

    def __init__(self, ctx, config: AFFConfig) -> None:
        # `ctx` is a metachase.session.Context, passed in (not imported here) to
        self.ctx = ctx
        self.config = config
        config.validate()  # fail fast on unimplemented knob values
        # Set by run() before chase dispatch; defaulted here so direct _chase_*
        self._run_t0 = 0.0
        self._two_phase_per_task = False

    # Public entry

    def run(self) -> AFFResult:
        """Execute the AFF system per the config and return an AFFResult."""
        t_total_start = time.perf_counter()
        self._run_t0 = t_total_start    # for relative P_out-event timestamps (B1)
        self._cum_steps_global = 0      # global total chasing-step offset for events

        tasks = self._select_tasks()

        # two_phase = evidence motifs first (in policy order), decision motifs
        force_canonicalize = False
        self._two_phase_per_task = False
        if self.config.decision_mode == "two_phase":
            if self.config.sharing != "shared":
                raise NotImplementedError(
                    "decision_mode='two_phase' requires sharing='shared'"
                )
            force_canonicalize = True
            if self.config.two_phase_scope == "task":
                # WITHIN-task two-phase (B5, paper Fig-4): run each task's own
                self._two_phase_per_task = True

        result = AFFResult(p_out=set(), config=self.config)

        if self.config.order_strategy == "dynamic_ready":
            if self.config.sharing != "shared":
                raise NotImplementedError("dynamic_ready requires sharing='shared'")
            self._chase_dynamic_ready(tasks, result)
        else:
            order = self._compute_order(tasks)
            if (self.config.decision_mode == "two_phase"
                    and self.config.two_phase_scope != "task"):
                # GLOBAL two-phase: evidence motifs first, decision motifs last.
                order = self._reorder_two_phase(tasks, order)
            self._validate_order(order, tasks)
            if self.config.sharing == "shared":
                self._chase_shared(tasks, order, result)
            elif self.config.sharing == "fresh":
                self._chase_fresh(tasks, order, result)
            else:
                raise NotImplementedError(f"sharing={self.config.sharing}")

        if (self.config.canonicalize or force_canonicalize) \
                and not self.config.no_final_canon:
            self._apply_canonicalize(result)

        result.timing["total"] = time.perf_counter() - t_total_start
        return result

    def _reorder_two_phase(self, tasks: List, order: List[int]) -> List[int]:
        """Split user's order into evidence/decision motifs and concatenate evid_first, dec_last (paper Sec V.C two-phase protocol)."""
        from ..confluence import gar_head_label

        p_out_ids = set(self.ctx.p_out_ids)

        def is_dec(ci: int) -> bool:
            gars = self._gars_of(tasks[ci])
            return any(gar_head_label(g) in p_out_ids for g in gars)

        evid_first = [ci for ci in order if not is_dec(ci)]
        dec_last = [ci for ci in order if is_dec(ci)]
        return evid_first + dec_last

    # Internals: task selection

    def _select_tasks(self) -> List:
        """Return the list of per-task GAR subsets."""
        ts = self.config.task_source
        if ts == "prebuilt":
            # Caller supplies the task list directly (e.g. build_block_type_tasks
            if self.config.prebuilt_tasks is None:
                raise ValueError("task_source='prebuilt' requires prebuilt_tasks")
            return list(self.config.prebuilt_tasks)
        if ts == "scc_blocks":
            return list(self.ctx.clusters_gars)
        if ts == "all_rules":
            # Single task = the full unsliced rule set (the classical FF baseline).
            return [self.ctx.gars_full]
        if ts == "all_rules_focused":
            # Single task = P_out-relevant rules (Meta-Chase upper bound on one shot).
            return [self.ctx.gars_focused]
        raise NotImplementedError(f"task_source={ts}")

    @staticmethod
    def _gars_of(task) -> List:
        """Uniform accessor: SCC blocks ARE the GAR list; enum-split is a dict."""
        if isinstance(task, dict):
            return task["gars"]
        return task

    def _validate_order(self, order: List[int], tasks: List) -> None:
        """Sanity-check that every task_idx in `order` is in range."""
        n = len(tasks)
        bad = [i for i in order if not (0 <= i < n)]
        if bad:
            raise IndexError(
                f"AFFConfig.custom_order contains task indices out of range "
                f"for task_source={self.config.task_source!r}: bad indices "
                f"{bad[:5]}{'...' if len(bad) > 5 else ''}; valid range [0, {n}). "
                f"This typically means caller asked for k > number of available "
                f"tasks."
            )

    def _compute_order(self, tasks: List) -> List[int]:
        """Return the task-index sequence."""
        cfg = self.config
        n = len(tasks)
        strat = cfg.order_strategy

        if strat == "natural":
            return list(range(n))
        if strat == "custom":
            if cfg.custom_order is None:
                raise ValueError("order_strategy='custom' requires custom_order")
            return list(cfg.custom_order)

        # FIFO goes through the policy dispatch. For scc_blocks the ctx ships
        if strat == "fifo":
            return self._policy_order(strat, n, tasks)

        raise NotImplementedError(f"order_strategy={strat}")

    def _task_meta_and_dag(self, tasks):
        """(clusters_meta, dag_edges) for the policy."""
        if self.config.task_source == "scc_blocks":
            return self.ctx.clusters_meta, self.ctx.block_dag
        from ..policy import task_dag_from_meta
        meta_by_id = {m["id"]: m for m in self.ctx.meta_full}
        clusters_meta = [[meta_by_id[g.id] for g in self._gars_of(t)
                          if g.id in meta_by_id] for t in tasks]
        return clusters_meta, task_dag_from_meta(clusters_meta)

    def _policy_order(self, strat: str, n: int, tasks: List) -> List[int]:
        """Compute order via a policy."""
        cfg = self.config
        from ..policy import FIFOPolicy, build_task_infos

        clusters_meta, dag = self._task_meta_and_dag(tasks)

        infos = build_task_infos(clusters_meta, self.ctx.cfg.p_out, dag_edges=dag)

        # Warm the policy's stats S from the driver's task-independent topo-pass
        if cfg.warmup_stats:
            for info in infos:
                st = cfg.warmup_stats.get(info.task_id)
                if st is not None:
                    g, c = st
                    info.stats.update(gain=float(g), noise=0.0, cost=float(c))

        if strat == "fifo":
            pol = FIFOPolicy()
        else:
            raise NotImplementedError(strat)
        return pol.order(infos, dag=dag)

    # Internals: dispatch — Phase 1 implements only 'shared' chase

    def _make_coord(self, gars: List) -> VanillaCoordinator:
        """Build a coordinator from the current config + ctx state."""
        cfg = self.config
        return VanillaCoordinator(
            self.ctx.graph,
            gars,
            max_rounds=cfg.max_rounds,
            max_matches=cfg.max_matches,
            max_derived=cfg.max_derived,
            verbose=cfg.verbose,
            matcher=self.ctx.matcher,
            p_out_ids=self.ctx.p_out_ids,
            functional_p_out=(self.ctx.functional_p_out_ids
                              if cfg.base_authority else None),
            early_commit=(cfg.decision_mode == "early_commit"),
            record_p_out_events=cfg.record_p_out_events,
        )

    def _chase_shared(self, tasks: List, order: List[int], result: AFFResult) -> None:
        """Run tasks in `order` against ONE shared (DerivedFacts, fired_pairs)."""
        cfg = self.config
        derived = DerivedFacts()
        fired: set = set()    # plain set() — matches legacy byte-for-byte
        per_task_timing: List[float] = []
        passes = 0
        n_tasks = len(order)
        # reinvoke_incremental: per-block overlay watermark (edge count at the end of
        block_wm: dict = {}
        reinc = cfg.reinvoke_incremental
        # Optional per-invocation trace (set runner._trace_edges=True before .run()).
        trace_on = getattr(self, "_trace_edges", False)
        if trace_on:
            self._trace = []

        while True:
            pass_new = 0
            for k, ci in enumerate(order, 1):
                t0 = time.perf_counter()
                # Task-level progress line (carriage-return, single-line refresh
                self._task_progress_tick(k, n_tasks, ci, tasks)
                task_id_tag = cfg.task_id_fmt.format(i=ci)
                seed_edges = None
                if reinc and ci in block_wm:
                    seed_edges = derived.edges_since(block_wm[ci])
                n_before = derived.num_edges
                res, prov, events, new_facts, run_stats = self._run_one_task(
                    self._gars_of(tasks[ci]), derived, fired, task_id_tag,
                    seed_edges=seed_edges)
                if trace_on:
                    self._trace.append({
                        "pass": passes + 1, "k": k, "block": ci,
                        "new_edges": derived.edges_since(n_before)})
                if reinc:
                    block_wm[ci] = derived.num_edges
                per_task_timing.append(time.perf_counter() - t0)

                result.all_results.extend(res)
                result.per_task.append((ci, run_stats))
                for triple, gids in prov.items():
                    result.provenance.setdefault(triple, set()).update(gids)
                result.p_out_event_log.extend(events)
                pass_new += new_facts
                if cfg.track_snapshots:
                    self._record_snapshot(result, ci)

            passes += 1
            if (not cfg.to_quiescence) or pass_new == 0 or passes >= cfg.max_passes:
                break

        result.p_out = confluence.collect_p_out(result.all_results, self.ctx.p_out_ids)
        result.timing["per_task"] = per_task_timing
        if getattr(self, "_trace_edges", False):
            self._final_overlay = set(derived._edge_set)   # full derived fact set

    def _chase_fresh(self, tasks: List, order: List[int], result: AFFResult) -> None:
        """Run each task with its OWN (DerivedFacts, fired_pairs) — no sharing."""
        cfg = self.config
        per_task_timing: List[float] = []
        union_pout: set = set()
        n_tasks = len(order)

        # 'fresh' doesn't iterate passes for quiescence — each task is

        for k, ci in enumerate(order, 1):
            t0 = time.perf_counter()
            self._task_progress_tick(k, n_tasks, ci, tasks)
            task_id_tag = cfg.task_id_fmt.format(i=ci)
            res, prov, events, _new, run_stats = self._run_one_task(
                self._gars_of(tasks[ci]), DerivedFacts(), set(), task_id_tag)
            per_task_timing.append(time.perf_counter() - t0)

            result.all_results.extend(res)
            result.per_task.append((ci, run_stats))
            for triple, gids in prov.items():
                result.provenance.setdefault(triple, set()).update(gids)
            result.p_out_event_log.extend(events)
            union_pout |= confluence.collect_p_out(res, self.ctx.p_out_ids)
            if cfg.track_snapshots:
                self._record_snapshot(result, ci, cumulative_pout=union_pout)

        result.p_out = union_pout
        result.timing["per_task"] = per_task_timing

    def _base_label_counts(self):
        """#base edges per label (cached, one-time)."""
        if not hasattr(self, "_base_lc_cache"):
            from collections import defaultdict
            lc: Dict[int, int] = defaultdict(int)
            for _node, adj in self.ctx.graph.out_adj.items():
                for (_dst, lbl) in adj:
                    lc[lbl] += 1
            self._base_lc_cache = lc
        return self._base_lc_cache

    def _chase_dynamic_ready(self, tasks: List, result: AFFResult) -> None:
        """Γ-aware DYNAMIC readiness scheduler (paper π(W,Γ), cheap variant)."""
        from collections import defaultdict
        from ..confluence import gar_head_label
        from ..graph.gar import ConclusionType
        cfg = self.config
        n = len(tasks)
        io = []                       # per task: (body_labels, gated_gars)
        for t in tasks:
            gars = self._gate(self._gars_of(t))
            body = set()
            for g in gars:
                for e in g.pattern.edges:
                    if e.label is not None:
                        body.add(e.label)
            io.append((body, gars))
        base_lc = self._base_label_counts()
        derived_lc: Dict[int, int] = defaultdict(int)

        def avail(l):
            return base_lc.get(l, 0) + derived_lc[l]

        prior = cfg.dynamic_yield_prior
        last_input = [0] * n
        derived = DerivedFacts()
        fired: set = set()
        per_task_timing: List[float] = []
        for _step in range(4 * n + 4):
            best_i, best_score = None, 0.0
            for i in range(n):
                body, gars = io[i]
                if not gars:
                    continue
                cur = sum(avail(l) for l in body)
                readiness = cur - last_input[i]
                if readiness <= 0:
                    continue
                pr = prior[i] if prior and i < len(prior) else 1.0
                score = readiness * pr
                if best_i is None or score > best_score:
                    best_i, best_score = i, score
            if best_i is None:
                break                  # no task's input grew -> fixpoint
            body, gars = io[best_i]
            t0 = time.perf_counter()
            coord = self._make_coord(gars)
            res = coord.run(existing_derived=derived, existing_fired_pairs=fired,
                            task_id=cfg.task_id_fmt.format(i=best_i))
            per_task_timing.append(time.perf_counter() - t0)
            for r in res:
                if r.conclusion == ConclusionType.ENRICH:
                    derived_lc[r.output[1]] += 1
            last_input[best_i] = sum(avail(l) for l in body)
            result.all_results.extend(res)
            result.per_task.append((best_i, coord.run_stats))
            for triple, gids in coord.p_out_provenance.items():
                result.provenance.setdefault(triple, set()).update(gids)
            result.p_out_event_log.extend(self._collect_events(coord))

        result.p_out = confluence.collect_p_out(result.all_results, self.ctx.p_out_ids)
        result.timing["per_task"] = per_task_timing

    # Internals: per-task execution (single coord, or within-task two-phase)

    def _gate(self, gars):
        """Optionally shuffle the firing order (shuffle_rules_seed) to remove the rule file's high-PCA-first bias that unfairly helps EC."""
        if self.config.shuffle_rules_seed is not None:
            import random as _r
            gars = list(gars)
            _r.Random(self.config.shuffle_rules_seed).shuffle(gars)
        return gars

    def _run_one_task(self, gars, derived, fired, task_id_tag, seed_edges=None):
        """Run ONE task's rules on the (possibly shared) overlay."""
        gars = self._gate(gars)
        if self._two_phase_per_task:
            return self._run_task_two_phase(gars, derived, fired, task_id_tag,
                                            seed_edges=seed_edges)
        coord = self._make_coord(gars)
        res = coord.run(existing_derived=derived, existing_fired_pairs=fired,
                        task_id=task_id_tag,
                        seed_delta_edges=seed_edges,
                        incremental=bool(seed_edges is not None))
        events = self._collect_events(coord)
        return (res, coord.p_out_provenance, events,
                getattr(coord.run_stats, "new_facts", 0), coord.run_stats)

    def _collect_events(self, coord):
        """Offset a coordinator's LOCAL P_out events to GLOBAL coordinates: time relative to the AFF run start, step_idx shifted by the chasing steps of all prior coordinators in this run."""
        base = self._cum_steps_global
        out = [(s, l, o, g, n, t - self._run_t0, base + step)
               for (s, l, o, g, n, t, step) in coord.p_out_events]
        self._cum_steps_global += coord.run_stats.chasing_steps
        return out

    def _run_task_two_phase(self, gars, derived, fired, task_id_tag, seed_edges=None):
        """WITHIN-task two-phase (paper Fig-4 TwoPhaseChase) on the shared overlay."""
        import os
        from ..confluence import split_evidence_decision, two_phase_alternate
        from ..chase.coordinator import RunStats
        inc_ok = os.environ.get("FF_TP_INCREMENTAL", "1") == "1"
        evid, dec = split_evidence_decision(gars, set(self.ctx.p_out_ids))
        all_res: List = []
        prov: Dict = {}
        events: List = []
        agg = RunStats(task_id=task_id_tag)

        def run_group(group, tag, seed_edges=None, seed_attrs=None, incremental=False):
            """Run one rule group on the shared overlay; accumulate; return its produced delta (edges, attr-names) so the caller can seed the other phase's re-run."""
            if not group:
                return [], set()
            coord = self._make_coord(group)
            r = coord.run(existing_derived=derived, existing_fired_pairs=fired,
                          task_id=tag, seed_delta_edges=seed_edges,
                          seed_delta_attrs=seed_attrs, incremental=incremental)
            all_res.extend(r)
            for triple, gids in coord.p_out_provenance.items():
                prov.setdefault(triple, set()).update(gids)
            events.extend(self._collect_events(coord))
            agg.chasing_steps += coord.run_stats.chasing_steps
            agg.rule_invocations += coord.run_stats.rule_invocations
            agg.new_facts += coord.run_stats.new_facts
            return coord.produced_edges, coord.produced_attrs

        # Protocol control flow lives in ONE place (confluence.two_phase_alternate) so
        two_phase_alternate(evid, dec, derived, run_group, inc_ok,
                            seed_edges=seed_edges, tag=task_id_tag)
        return all_res, prov, events, agg.new_facts, agg

    # Internals: per-task progress line (stdout \r refresh; coord tqdm

    _PROGRESS_STREAM = sys.stdout

    def _task_progress_tick(self, k: int, n: int, task_idx, tasks) -> None:
        """Print one self-overwriting progress line to stdout each task."""
        payload = tasks[task_idx]
        if isinstance(payload, dict):
            label = payload.get("label") or payload.get("predicate") or str(task_idx)
        else:
            label = f"block-{task_idx}"
        # Truncate long labels so the line fits.
        if len(label) > 32:
            label = label[:29] + "..."
        msg = f"  ▸ task {k:>3}/{n}  {label:<32}"
        self._PROGRESS_STREAM.write("\r" + msg)
        self._PROGRESS_STREAM.flush()
        if k == n:
            self._PROGRESS_STREAM.write("\n")
            self._PROGRESS_STREAM.flush()

    # Internals: per-task snapshot capture (for shift / coverage curves)

    def _record_snapshot(self, result: AFFResult, task_idx,
                         cumulative_pout: Optional[Set[Triple]] = None) -> None:
        """Append one snapshot dict to result.per_task_snapshots."""
        if cumulative_pout is None:
            cumulative_pout = confluence.collect_p_out(result.all_results, self.ctx.p_out_ids)
        snap = {
            "task_idx": task_idx,
            "cumulative_p_out": set(cumulative_pout),       # COPY (caller may mutate later)
            "n_facts_so_far": len(cumulative_pout),
        }
        if self.config.canonicalize:
            canon_set = confluence.canonicalize(
                result.all_results, self.ctx.p_out_ids, self.ctx.pca_by_gar,
                functional=self.ctx.functional_p_out, provenance=result.provenance,
            )
            snap["cumulative_canonical"] = {(s, p): o for (s, p, o) in canon_set}
        result.per_task_snapshots.append(snap)

    # Internals: post-processing

    def _apply_canonicalize(self, result: AFFResult) -> None:
        """Apply max-PCA canonicalization to the raw fire set (Sec V.B)."""
        canon_set = confluence.canonicalize(
            result.all_results,
            self.ctx.p_out_ids,
            self.ctx.pca_by_gar,
            functional=self.ctx.functional_p_out,
            provenance=result.provenance,
        )
        # canon_set is Set[Triple]; expose both shapes — keep .p_out as the
        result.p_out = set(canon_set)
        result.canonical_decisions = {(s, p): o for (s, p, o) in canon_set}
