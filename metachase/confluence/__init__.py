"""metachase.confluence — output-consistency mechanisms (paper Section V)."""

from .canon import (
    gar_head_label,
    collect_p_out,
    decide,
    canonicalize,
    _candidates_by_key,
    _is_functional,
    Triple,
)
from .two_phase import (
    split_evidence_decision,
    two_phase_alternate,
    two_phase_chase,
    _dependency_strata,
    _evidence_consumes_p_out,
    _body_labels,
    _collect_p_out_from_derived,
)
from .community import (
    check_output_commutativity,
    check_community_exact,
)

__all__ = [
    # V.B canonicalization + decision logic
    "gar_head_label", "collect_p_out", "decide", "canonicalize",
    # V.C two-phase
    "split_evidence_decision", "two_phase_alternate", "two_phase_chase",
    # V.A community check
    "check_output_commutativity", "check_community_exact",
]
