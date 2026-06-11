"""Graph Association Rule (GAR) data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple


# Pattern graph

@dataclass
class PatternNode:
    """A node variable in the pattern graph."""
    var: str
    label: Optional[int] = None       # None ↔ wildcard


@dataclass
class PatternEdge:
    """A directed edge in the pattern graph."""
    src_var: str
    dst_var: str
    label: Optional[int] = None       # None ↔ wildcard


@dataclass
class PatternGraph:
    """The structural part Q[x̄] of a GAR."""
    nodes: Dict[str, PatternNode]
    edges: List[PatternEdge]

    def neighbours_of(self, var: str) -> List[Tuple[str, int | None, str]]:
        """Return (peer_var, edge_label, direction) for all edges incident on *var*."""
        result = []
        for e in self.edges:
            if e.src_var == var:
                result.append((e.dst_var, e.label, "out"))
            elif e.dst_var == var:
                result.append((e.src_var, e.label, "in"))
        return result


# Predicates

class PredicateKind(Enum):
    LITERAL = auto()    # attribute comparison, degree check, …
    ML      = auto()    # learned function predicate (Phase 5)


@dataclass
class Predicate:
    """A single predicate φᵢ in the GAR body."""
    kind:             PredicateKind
    fn:               Callable[[Dict[str, int], Any], bool]
    description:      str       = ""
    referenced_attrs: Set[str]  = field(default_factory=set)

    @property
    def has_unknown_attr_deps(self) -> bool:
        """True when the predicate has opaque attribute dependencies."""
        return "*" in self.referenced_attrs

    # --- convenience constructors -------------------------------------------

    @staticmethod
    def degree_ge(var: str, k: int) -> "Predicate":
        """Degree of matched node for *var* must be ≥ k (out-degree)."""
        def _check(match, graph):
            nid = match[var]
            return graph.out_degree(nid) >= k
        return Predicate(
            kind=PredicateKind.LITERAL,
            fn=_check,
            description=f"degree({var}) >= {k}",
            # degree is structural, not attribute-based
        )

    @staticmethod
    def not_same(var1: str, var2: str) -> "Predicate":
        """The two matched nodes must be distinct (anti-reflexivity)."""
        def _check(match, graph):
            return match[var1] != match[var2]
        return Predicate(
            kind=PredicateKind.LITERAL,
            fn=_check,
            description=f"{var1} ≠ {var2}",
        )

    @staticmethod
    def attr_eq(var: str, attr: str, value: str) -> "Predicate":
        """Attribute of matched node equals a constant: ``x.attr == value``."""
        def _check(match, graph):
            return graph.get_attr(match[var], attr) == value
        return Predicate(
            kind=PredicateKind.LITERAL,
            fn=_check,
            description=f"{var}.{attr} == {value!r}",
            referenced_attrs={attr},
        )

    @staticmethod
    def attr_eq_attr(var1: str, attr1: str,
                     var2: str, attr2: str) -> "Predicate":
        """Two matched nodes share the same attribute value: ``x.attr1 == y.attr2``."""
        def _check(match, graph):
            v1 = graph.get_attr(match[var1], attr1)
            v2 = graph.get_attr(match[var2], attr2)
            return v1 is not None and v1 == v2
        return Predicate(
            kind=PredicateKind.LITERAL,
            fn=_check,
            description=f"{var1}.{attr1} == {var2}.{attr2}",
            referenced_attrs={attr1, attr2},
        )

    @staticmethod
    def attr_contains(var: str, attr: str, substring: str) -> "Predicate":
        """Attribute value contains *substring* (case-insensitive)."""
        sub_lower = substring.lower()
        def _check(match, graph):
            val = graph.get_attr(match[var], attr)
            return val is not None and sub_lower in val.lower()
        return Predicate(
            kind=PredicateKind.LITERAL,
            fn=_check,
            description=f"{var}.{attr} contains {substring!r}",
            referenced_attrs={attr},
        )

    @staticmethod
    def attr_exists(var: str, attr: str) -> "Predicate":
        """Node has *attr* set (is not absent)."""
        def _check(match, graph):
            return graph.get_attr(match[var], attr) is not None
        return Predicate(
            kind=PredicateKind.LITERAL,
            fn=_check,
            description=f"{var}.{attr} exists",
            referenced_attrs={attr},
        )

    @staticmethod
    def attr_ne(var: str, attr: str, value: str) -> "Predicate":
        """Attribute of matched node is NOT equal to a constant: ``x.attr ≠ value``."""
        def _check(match, graph):
            v = graph.get_attr(match[var], attr)
            return v is not None and v != value
        return Predicate(
            kind=PredicateKind.LITERAL,
            fn=_check,
            description=f"{var}.{attr} ≠ {value!r}",
            referenced_attrs={attr},
        )

    @staticmethod
    def attr_cmp(var: str, attr: str, op: str,
                 value: float | int) -> "Predicate":
        """Numeric comparison on an attribute: ``x.attr op value``."""
        import operator as _op
        _ops = {
            "<":  _op.lt, "<=": _op.le,
            ">":  _op.gt, ">=": _op.ge,
        }
        if op not in _ops:
            raise ValueError(f"op must be one of {list(_ops)}; got {op!r}")
        fn_op = _ops[op]

        def _check(match, graph):
            raw = graph.get_attr(match[var], attr)
            if raw is None:
                return False
            try:
                return fn_op(float(raw), float(value))
            except (ValueError, TypeError):
                return False
        return Predicate(
            kind=PredicateKind.LITERAL,
            fn=_check,
            description=f"{var}.{attr} {op} {value}",
            referenced_attrs={attr},
        )

    @staticmethod
    def custom(fn: Callable, description: str = "",
               referenced_attrs: Optional[Set[str]] = None) -> "Predicate":
        """Wrap an arbitrary boolean function as a LITERAL predicate."""
        return Predicate(
            kind=PredicateKind.LITERAL,
            fn=fn,
            description=description,
            referenced_attrs=referenced_attrs if referenced_attrs is not None else {"*"},
        )


# Conclusion

class ConclusionType(Enum):
    DEDUCE  = auto()    # fill in a missing attribute value
    ENRICH  = auto()    # assert a new edge between two matched nodes
    CLEAN   = auto()    # flag a node / edge as a potential error


@dataclass
class Conclusion:
    """The consequent ψ(x, y) of a GAR."""
    type:   ConclusionType
    target: Any
    label:  str = ""    # human-readable label shown in output rows

    # --- convenience constructors -------------------------------------------

    @staticmethod
    def enrich_edge(src_var: str, edge_label_id: int, dst_var: str,
                    label: str = "") -> "Conclusion":
        return Conclusion(type=ConclusionType.ENRICH,
                          target=(src_var, edge_label_id, dst_var),
                          label=label)

    @staticmethod
    def deduce_attr(var: str, attr: str, value: Any,
                    label: str = "") -> "Conclusion":
        return Conclusion(type=ConclusionType.DEDUCE,
                          target=(var, attr, value), label=label)

    @staticmethod
    def clean_node(var: str, label: str = "") -> "Conclusion":
        return Conclusion(type=ConclusionType.CLEAN,
                          target=(var,), label=label)


# GAR

@dataclass
class GAR:
    """A single Graph Association Rule."""
    id:          str
    pattern:     PatternGraph
    predicates:  List[Predicate]
    conclusion:  Conclusion
    description: str = ""

    def check_predicates(self, match: Dict[str, int], graph: Any) -> bool:
        """Return True iff all body predicates hold for *match*."""
        return all(p.fn(match, graph) for p in self.predicates)
