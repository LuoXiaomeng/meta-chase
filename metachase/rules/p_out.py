"""Output predicate sets (P_out) for Meta-Chase confluence analysis."""

from __future__ import annotations

from typing import Set


# DBpedia: predicates suitable as output / DECISION facts.
DBPEDIA_P_OUT: Set[str] = {
    'owner',           # synonym cycle (owner<->owningCompany), ~2 rounds, 2 rules
    'stateOfOrigin',   # synonym cycle (<->nationality), incl. R0359 multi-evidence
    'homeStadium',     # synonym cycle (<->ground), ~2 rounds
    'locationCountry', # acyclic, 3 competing rules
    'director',        # acyclic, 2 multi-evidence rules (editing/cinematography & writer)
}

# Evidence rules that explode (deep multi-atom cycles) but are NOT propagation
DBPEDIA_CYCLE_BREAKERS = [
    'R0451',  # city & countySeat -> county   (county<->countySeat 2-cycle)
    'R0221',  # city & county -> countySeat
]

# Rules excluded from the experimental rule set.
DBPEDIA_EXCLUDED_RULES = [
    # (a) High-support propagation (sup >= 30K)
    'R0553',  # team        sup=1,218,883
    'R0523',  # country     sup=338,152
    'R0552',  # subdivision sup=64,507
    'R0551',  # order       sup=43,860
    'R0524',  # country     sup=38,275

    # (b) Smaller-support but explosive in chase (growth >= 1.8x/round)
    'R0283',  # sire & sire -> grandsire (self-feedback)
    'R0286',  # series
    'R0296',  # domain
    'R0297',  # domain
    'R0299',  # domain
    'R0300',  # domain
    'R0317',  # division
    'R0335',  # region
    'R0389',  # state
    'R0451',  # county
    'R0522',  # country
    'R0544',  # openingTheme

    # (c) Discovered after extending to non-path rules (full 119-rule set)
    'R0263',  # city & federalState -> federalState (propagation)
    'R0392',  # county & state -> state (cross-pred propagation)
    'R0463',  # previousWork & type -> musicType (indirect cycle via R0072/R0135)
    'R0467',  # subsequentWork & type -> musicType (indirect cycle via R0072/R0135)
]

# Predicates that should NEVER be in P_out for DBpedia
DBPEDIA_EVIDENCE_ONLY: Set[str] = {
    'locatedIn',    # transitive closure explodes
    'type',         # subclass propagation
    'subClassOf',
    'seeAlso',      # symmetric, growth
}


def get_p_out(dataset: str) -> Set[str]:
    """Return the output predicate set for a given dataset."""
    try:
        from .dataset_config import get_config
        return set(get_config(dataset).p_out)
    except Exception:
        if dataset == 'dbpedia':
            return DBPEDIA_P_OUT
        raise ValueError(f"No P_out defined for dataset: {dataset}")


def get_evidence_only(dataset: str) -> Set[str]:
    """Return predicates that must be excluded from P_out."""
    if dataset == 'dbpedia':
        return DBPEDIA_EVIDENCE_ONLY
    return set()


def get_excluded_rules(dataset: str) -> list:
    """Return rule IDs to exclude IN ADDITION to structural propagation filtering."""
    try:
        from .dataset_config import get_config
        return list(get_config(dataset).cycle_breaker_rules)
    except Exception:
        if dataset == 'dbpedia':
            return list(DBPEDIA_CYCLE_BREAKERS)
        return []


def resolve_p_out_ids(p_out_names: Set[str], edge_label_dict: dict) -> Set[int]:
    """Convert predicate names to edge label IDs, dropping any not in the graph."""
    ids = set()
    missing = []
    for name in p_out_names:
        if name in edge_label_dict:
            ids.add(edge_label_dict[name])
        else:
            missing.append(name)
    if missing:
        print(f"[p_out] WARNING: predicates not found in graph: {missing}")
    return ids
