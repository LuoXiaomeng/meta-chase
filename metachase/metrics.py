"""Evaluation metrics for Meta-Chase experiments."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

Triple = Tuple[int, int, int]


# Loading

def load_ground_truth(path: Path) -> dict:
    """Load a pickled ground-truth dict from build_ground_truth.py."""
    with open(path, "rb") as f:
        return pickle.load(f)


# Core metrics

def precision(predicted: Set[Triple], reference: Set[Triple]) -> float:
    """|pred ∩ ref| / |pred|"""
    if not predicted:
        return 0.0
    return len(predicted & reference) / len(predicted)


def recall(predicted: Set[Triple], reference: Set[Triple]) -> float:
    """|pred ∩ ref| / |ref|"""
    if not reference:
        return 0.0
    return len(predicted & reference) / len(reference)


def f1(predicted: Set[Triple], reference: Set[Triple]) -> float:
    """Harmonic mean of precision and recall."""
    p = precision(predicted, reference)
    r = recall(predicted, reference)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def jaccard(a: Set[Triple], b: Set[Triple]) -> float:
    """|a ∩ b| / |a ∪ b|."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# Aggregate / multi-schedule metrics

def pairwise_jaccard_matrix(fact_sets: List[Set[Triple]]) -> List[List[float]]:
    """For N schedule outputs, return N×N matrix of pairwise Jaccard similarity."""
    n = len(fact_sets)
    mat = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = jaccard(fact_sets[i], fact_sets[j])
            mat[i][j] = mat[j][i] = v
    return mat


def consistency_rate(fact_sets: List[Set[Triple]]) -> Tuple[float, int]:
    """How many distinct outputs across schedules."""
    if not fact_sets:
        return 1.0, 0
    unique = {frozenset(s) for s in fact_sets}
    n_unique = len(unique)
    rate = 1.0 if n_unique == 1 else 1.0 / n_unique
    return rate, n_unique


def mean_pairwise_jaccard(fact_sets: List[Set[Triple]]) -> float:
    """Average off-diagonal Jaccard."""
    n = len(fact_sets)
    if n < 2:
        return 1.0
    total = 0.0
    cnt   = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += jaccard(fact_sets[i], fact_sets[j])
            cnt   += 1
    return total / max(cnt, 1)


# Convenience: filter result iterable to P_out triples

def extract_triples(results: Iterable, p_out_ids: Set[int] = None) -> Set[Triple]:
    """Pull (src, label, dst) triples from a list of FiringResult."""
    out = set()
    for r in results:
        o = getattr(r, 'output', None)
        if isinstance(o, tuple) and len(o) == 3:
            s, l, d = o
            if p_out_ids is None or l in p_out_ids:
                out.add((s, l, d))
    return out


# Report

def print_quality_table(
    label_to_facts: Dict[str, Set[Triple]],
    reference: Set[Triple],
    title: str = "Quality vs Ground Truth",
) -> None:
    """Print a precision/recall/F1 table for multiple methods vs one reference."""
    print(f"\n  {title}")
    print(f"  {'-'*70}")
    print(f"  {'Method':25s}  {'|facts|':>8}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Jaccard':>7}")
    print(f"  {'-'*70}")
    for label, facts in label_to_facts.items():
        p = precision(facts, reference)
        r = recall(facts, reference)
        f = f1(facts, reference)
        j = jaccard(facts, reference)
        print(f"  {label:25s}  {len(facts):>8,}  {p:>6.3f}  {r:>6.3f}  "
              f"{f:>6.3f}  {j:>7.3f}")
    print(f"  Reference: {len(reference):,} facts")
