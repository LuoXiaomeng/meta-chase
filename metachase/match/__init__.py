"""metachase.match — subgraph-isomorphism matching (full + delta)."""
from .matcher import Matcher
from .delta import match_delta

__all__ = ["Matcher", "match_delta"]
