"""
graph.py
--------
Shared weighted directed graph used by yen.py, sb.py, and visualizer.py.

Public API
~~~~~~~~~~
  Graph.add_node(x, y, label=None)  -> Node
  Graph.add_edge(u, v, w)
  Graph.remove_node(nid)
  Graph.remove_edge(idx)
  Graph.node_by_id(nid)             -> Node | None
  Graph.neighbors(nid)              -> list[(v, w)]
  Graph.adj(blocked_nodes, blocked_edges) -> dict[id -> list[(v,w)]]
  Graph.reverse_adj()               -> dict[id -> list[(u,w)]]

  dijkstra(adj, source, all_ids)    -> (dist, prev)
  reconstruct(prev, src, tgt)       -> list[int] | None
  path_cost(g, path)                -> float
"""

from __future__ import annotations
import heapq
import math
from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class Node:
    id: int
    x: float
    y: float
    label: str


@dataclass
class Edge:
    u: int
    v: int
    w: float


# ──────────────────────────────────────────────────────────────
# Graph
# ──────────────────────────────────────────────────────────────

class Graph:
    """Adjacency-list weighted directed graph with pygame-friendly node positions."""

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self._nid = 0

    # ── construction ──────────────────────────────────────────

    def add_node(self, x: float, y: float, label: str | None = None) -> Node:
        lbl = label or self._next_label()
        n = Node(self._nid, x, y, lbl)
        self._nid += 1
        self.nodes.append(n)
        return n

    def _next_label(self) -> str:
        used = {n.label for n in self.nodes}
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if c not in used:
                return c
        return str(self._nid)

    def add_edge(self, u: int, v: int, w: float) -> None:
        self.edges.append(Edge(u, v, w))

    def remove_node(self, nid: int) -> None:
        self.nodes = [n for n in self.nodes if n.id != nid]
        self.edges = [e for e in self.edges if e.u != nid and e.v != nid]

    def remove_edge(self, idx: int) -> None:
        if 0 <= idx < len(self.edges):
            self.edges.pop(idx)

    # ── queries ───────────────────────────────────────────────

    def node_by_id(self, nid: int) -> Optional[Node]:
        return next((n for n in self.nodes if n.id == nid), None)

    def neighbors(self, nid: int) -> list[tuple[int, float]]:
        """Return [(v, w), ...] for all outgoing edges from nid."""
        return [(e.v, e.w) for e in self.edges if e.u == nid]

    def all_ids(self) -> list[int]:
        return [n.id for n in self.nodes]

    def adj(
        self,
        blocked_nodes: set[int] | None = None,
        blocked_edges: set[tuple[int, int]] | None = None,
    ) -> dict[int, list[tuple[int, float]]]:
        """
        Build adjacency dict, optionally excluding certain nodes/edges.
        Used by Yen's algorithm to build restricted sub-graphs.
        """
        bn = blocked_nodes or set()
        be = blocked_edges or set()
        a: dict[int, list] = {n.id: [] for n in self.nodes}
        for e in self.edges:
            if e.u in bn or e.v in bn:
                continue
            if (e.u, e.v) in be:
                continue
            a[e.u].append((e.v, e.w))
        return a

    def reverse_adj(self) -> dict[int, list[tuple[int, float]]]:
        """Return adjacency dict with all edge directions reversed."""
        a: dict[int, list] = {n.id: [] for n in self.nodes}
        for e in self.edges:
            a[e.v].append((e.u, e.w))
        return a

    def __repr__(self) -> str:
        return f"Graph(nodes={len(self.nodes)}, edges={len(self.edges)})"


# ──────────────────────────────────────────────────────────────
# Shared algorithm utilities
# ──────────────────────────────────────────────────────────────

def dijkstra(
    adj: dict[int, list[tuple[int, float]]],
    source: int,
    all_ids: list[int],
) -> tuple[dict[int, float], dict[int, int]]:
    """
    Standard Dijkstra on an explicit adjacency dict.

    Returns
    -------
    dist : {node_id -> shortest distance from source}
    prev : {node_id -> predecessor node_id on shortest path}
    """
    dist: dict[int, float] = {nid: math.inf for nid in all_ids}
    prev: dict[int, int] = {}
    dist[source] = 0.0
    pq: list[tuple[float, int]] = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, prev


def reconstruct(
    prev: dict[int, int],
    src: int,
    tgt: int,
) -> list[int] | None:
    """Walk predecessor map backwards to rebuild a node-id path."""
    path: list[int] = []
    c: int | None = tgt
    while c is not None:
        path.append(c)
        if c == src:
            break
        c = prev.get(c)
    path.reverse()
    return path if (path and path[0] == src) else None


def path_cost(g: Graph, path: list[int]) -> float:
    """Sum edge weights along a node-id path."""
    lut = {(e.u, e.v): e.w for e in g.edges}
    return sum(lut.get((path[i], path[i + 1]), 0.0) for i in range(len(path) - 1))
