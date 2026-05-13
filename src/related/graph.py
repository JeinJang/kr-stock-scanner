from __future__ import annotations

import networkx as nx

from src.related.db import StoredEdge


def virtual_node(name: str) -> str:
    """Stable id for an unlisted/foreign company."""
    return f"_unlisted_{name}"


def build_graph(edges: list[StoredEdge]) -> nx.DiGraph:
    """Build a DiGraph from stored edges."""
    g = nx.DiGraph()
    for e in edges:
        target = e.target_ticker if e.target_ticker else virtual_node(e.target_name)
        g.add_node(e.source_ticker)
        g.add_node(target, display_name=e.target_name)
        g.add_edge(
            e.source_ticker, target,
            relation=e.relation, evidence=e.evidence,
        )
    return g


def expand(graph: nx.DiGraph, root: str, depth: int = 1) -> nx.DiGraph:
    """Return subgraph reachable from root within `depth` hops (both directions)."""
    if root not in graph:
        return graph.subgraph([]).copy()
    visited = {root}
    frontier = {root}
    for _ in range(depth):
        next_frontier = set()
        for node in frontier:
            next_frontier |= set(graph.successors(node))
            next_frontier |= set(graph.predecessors(node))
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break
    return graph.subgraph(visited).copy()
