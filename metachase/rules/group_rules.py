"""Group rules into independent closure clusters."""

from __future__ import annotations

import sys
from collections import defaultdict


# Backward relevance: the minimal rule subset needed to compute P_out

def relevant_rules(rules, p_out_preds):
    """Rules whose output can affect a P_out predicate (backward reachability)."""
    relevant_preds = set(p_out_preds)
    changed = True
    while changed:
        changed = False
        for r in rules:
            if r["head_atom"][1] in relevant_preds:
                for a in r["body_atoms"]:
                    if a[1] not in relevant_preds:
                        relevant_preds.add(a[1]); changed = True
    return [r for r in rules if r["head_atom"][1] in relevant_preds]


# SCC-only clustering (the design we converged on)

def producer_consumer_edges(rules):
    """Directed edges i->j iff head(rule_i) is a body predicate of rule_j (i!=j)."""
    producers = defaultdict(list)            # predicate -> [rule index producing it]
    for idx, r in enumerate(rules):
        producers[r["head_atom"][1]].append(idx)
    edges = set()
    for j, r in enumerate(rules):
        for bp in {a[1] for a in r["body_atoms"]}:
            for i in producers.get(bp, ()):
                if i != j:
                    edges.add((i, j))
    return edges


def _scc_tarjan(n, adj):
    """Tarjan's SCC."""
    index_counter = [0]
    stack, on_stack = [], [False] * n
    index, lowlink = [None] * n, [0] * n
    result = []
    sys.setrecursionlimit(1_000_000)

    def strongconnect(v):
        index[v] = lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v); on_stack[v] = True
        for w in adj[v]:
            if index[w] is None:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            comp = []
            while True:
                w = stack.pop(); on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            result.append(comp)

    for v in range(n):
        if index[v] is None:
            strongconnect(v)
    return result


def _toposort(n_nodes, edges):
    """Kahn topological sort (deterministic via sorted tie-breaks)."""
    from collections import deque
    indeg = [0] * n_nodes
    succ = defaultdict(list)
    for a, b in edges:
        succ[a].append(b); indeg[b] += 1
    q = deque(sorted(i for i in range(n_nodes) if indeg[i] == 0))
    order = []
    while q:
        u = q.popleft(); order.append(u)
        for v in sorted(succ[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order


def compute_scc_blocks(rules):
    """SCC-only clustering."""
    n = len(rules)
    if n == 0:
        return [], set()
    edges = producer_consumer_edges(rules)
    adj = defaultdict(list)
    for i, j in edges:
        adj[i].append(j)

    sccs = _scc_tarjan(n, adj)                       # reverse-topo order
    scc_of = {idx: sid for sid, comp in enumerate(sccs) for idx in comp}

    cond_edges = {(scc_of[i], scc_of[j]) for (i, j) in edges if scc_of[i] != scc_of[j]}
    topo = _toposort(len(sccs), cond_edges)          # sources first
    pos = {sid: p for p, sid in enumerate(topo)}     # scc-id -> topo position

    blocks = [None] * len(sccs)
    for sid, comp in enumerate(sccs):
        blocks[pos[sid]] = [rules[idx] for idx in comp]
    dag_edges = {(pos[a], pos[b]) for (a, b) in cond_edges}
    return blocks, dag_edges
