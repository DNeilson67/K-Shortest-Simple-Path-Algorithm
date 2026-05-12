"""
sb.py
-----
Suurballe–Bhandari (SB) K-Shortest Simple Paths algorithm.

References
~~~~~~~~~~
  Suurballe, J. W. (1974). Disjoint paths in a network.
    Networks, 4(2), 125–145.

  Bhandari, R. (1999). Survivable Networks: Algorithms for Diverse Routing.
    Kluwer Academic Publishers.

Algorithm overview
~~~~~~~~~~~~~~~~~~
The SB algorithm finds k shortest *edge-disjoint* simple paths from s to t
using an iterative graph-transformation approach:

  Iteration 1:
    Run Dijkstra(G, s) to get dist[] and a shortest-path tree T.

  For i = 1 … k-1:
    a. Transform the graph:
         - For every edge (u→v, w) on the current solution paths,
           replace it with its reverse (v→u) with weight  -w + dist[u] - dist[v]
           (i.e. zero-cost or negative on the shortest-path DAG, ensuring
           the modified Dijkstra still works via reduced costs).
    b. Run Dijkstra on the transformed graph from s → t.
         This finds an "augmenting path" that may share nodes with
         existing paths but uses the edge-reversal trick to implicitly
         cancel edges and produce new simple paths.
    c. Combine the augmented edge set with the previous paths.
         Extract k edge-disjoint simple paths by following edge directions
         from s to t, removing "interlacing" reversed edges.

Simplicity guarantee
~~~~~~~~~~~~~~~~~~~~
  After extraction, interlaced edges (forward+backward on same segment)
  cancel each other. The resulting paths are simple (no repeated nodes).

Complexity (theoretical)
~~~~~~~~~~~~~~~~~~~~~~~~
  Let n = |V|, m = |E|

  Time  : O(k · (m + n log n))
            k   iterations (one Dijkstra per new path)
          × O(m + n log n)  Dijkstra on the transformed graph

            This is significantly better than Yen's O(k·n·(m+n log n))
            because there is NO inner loop over spur nodes.

  Space : O(k · n + m)
            O(m) for the transformed graph edges at each step
          + O(k · n) for storing k paths of length ≤ n

Usage
~~~~~
  from graph import Graph
  from sb import sb_k_shortest

  result = sb_k_shortest(graph, source, target, k)
  # result: list of (path: list[int], cost: float, stats: dict)
"""

from __future__ import annotations
import heapq
import math
import time
import tracemalloc
from collections import defaultdict
from graph import Graph, dijkstra, path_cost


# ──────────────────────────────────────────────────────────────
# Complexity stats container
# ──────────────────────────────────────────────────────────────

def _empty_stats() -> dict:
    return {
        # Counters
        "dijkstra_calls":     0,   # Dijkstra invocations (one per iteration)
        "graph_transforms":   0,   # edge-reversal transform steps
        "edges_reversed":     0,   # total edges reversed across all iterations
        "paths_found":        0,   # confirmed paths returned
        # Complexity labels
        "time_complexity":    "O(k · (m + n log n))",
        "space_complexity":   "O(k · n + m)",
        # Measured
        "elapsed_ms":         0.0,
        "peak_memory_kb":     0,
    }


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _reduced_cost(w: float, u: int, v: int, dist: dict[int, float]) -> float:
    """
    Johnson's price-function reduced cost for edge (u→v, w).
    Ensures all reduced costs are non-negative so Dijkstra works
    even after negative-weight edge reversals.

      w_reduced(u,v) = w(u,v) + dist[u] - dist[v]
    """
    du = dist.get(u, math.inf)
    dv = dist.get(v, math.inf)
    if math.isinf(du) or math.isinf(dv):
        return math.inf
    return w + du - dv


def _build_transformed_graph(
    all_ids: list[int],
    base_edges: list[tuple[int, int, float]],   # (u, v, w)
    used_edges: set[tuple[int, int]],            # edges on current paths
    dist: dict[int, float],
    stats: dict,
) -> dict[int, list[tuple[int, float]]]:
    """
    Build the transformed adjacency dict for one SB iteration:

    - Edges NOT on any current path: keep with reduced cost.
    - Edges ON a current path (u→v): replace with reverse (v→u)
      using reduced cost  = 0  (because they are shortest-path edges,
      so dist[v] = dist[u] + w, hence reduced cost = w + dist[u] - dist[v] = 0).
      Practically we use -w_reduced to allow cancellation.

    This is the key transformation that allows the next Dijkstra to
    find an augmenting path that "cancels" segments of existing paths.
    """
    adj: dict[int, list[tuple[int, float]]] = {nid: [] for nid in all_ids}

    for (u, v, w) in base_edges:
        rc = _reduced_cost(w, u, v, dist)
        if math.isinf(rc):
            continue

        if (u, v) in used_edges:
            # Reverse the edge; reversed reduced cost = -rc (can be 0 or tiny float)
            rev_rc = max(0.0, -rc)          # clamp to 0 for numerical safety
            adj[v].append((u, rev_rc))
            stats["edges_reversed"] += 1
        else:
            adj[u].append((v, max(0.0, rc)))

    return adj


def _extract_paths(
    all_ids: list[int],
    edge_pool: dict[tuple[int, int], int],    # (u,v) -> multiplicity
    src: int,
    tgt: int,
    k: int,
) -> list[list[int]]:
    """
    Extract up to k simple paths from the multi-edge pool produced
    by k rounds of the SB augmentation.

    Strategy
    --------
    Edges in `edge_pool` represent the union of all augmented paths.
    Forward+backward copies of the same edge cancel (they are removed).
    After cancellation, trace paths greedily from src to tgt.
    """
    # Cancel interlaced edges (forward and backward cancel each other).
    fwd: dict[tuple[int, int], int] = defaultdict(int)
    for (u, v), cnt in edge_pool.items():
        if cnt > 0:
            fwd[(u, v)] += cnt

    # Remove any (u,v)+(v,u) pairs.
    to_cancel = []
    for (u, v), cnt in list(fwd.items()):
        rev_cnt = fwd.get((v, u), 0)
        if rev_cnt > 0:
            cancel = min(cnt, rev_cnt)
            to_cancel.append(((u, v), (v, u), cancel))
    for (e1, e2, c) in to_cancel:
        fwd[e1] -= c
        fwd[e2] -= c

    # Build final adjacency (remaining edges).
    out_adj: dict[int, list[int]] = defaultdict(list)
    for (u, v), cnt in fwd.items():
        for _ in range(cnt):
            out_adj[u].append(v)

    # Greedily trace paths from src to tgt.
    paths: list[list[int]] = []
    for _ in range(k):
        path = _trace(out_adj, src, tgt)
        if path is None:
            break
        # Remove used edges from pool.
        for i in range(len(path) - 1):
            out_adj[path[i]].remove(path[i + 1])
        paths.append(path)

    return paths


def _trace(
    adj: dict[int, list[int]],
    src: int,
    tgt: int,
) -> list[int] | None:
    """DFS to trace one simple path from src to tgt through remaining edges."""
    visited: set[int] = set()
    stack: list[tuple[int, list[int]]] = [(src, [src])]
    while stack:
        u, path = stack.pop()
        if u == tgt:
            return path
        if u in visited:
            continue
        visited.add(u)
        for v in adj.get(u, []):
            if v not in visited:
                stack.append((v, path + [v]))
    return None


# ──────────────────────────────────────────────────────────────
# Main algorithm
# ──────────────────────────────────────────────────────────────

def sb_k_shortest(
    g: Graph,
    src: int,
    tgt: int,
    k: int,
) -> list[tuple[list[int], float, dict]]:
    """
    Find up to k shortest simple (edge-disjoint) paths from src to tgt
    using the Suurballe–Bhandari algorithm.

    Parameters
    ----------
    g   : Graph
    src : source node id
    tgt : target node id
    k   : number of paths requested

    Returns
    -------
    List of (path, cost, stats) tuples ordered by cost ascending.
    stats is a shared dict with timing, memory, and operation counts.
    """
    stats = _empty_stats()
    tracemalloc.start()
    t0 = time.perf_counter()

    all_ids = g.all_ids()
    base_edges = [(e.u, e.v, e.w) for e in g.edges]

    # ── Iteration 0: initial Dijkstra ─────────────────────────
    adj0 = {nid: [] for nid in all_ids}
    for (u, v, w) in base_edges:
        adj0[u].append((v, w))

    dist, prev = dijkstra(adj0, src, all_ids)
    stats["dijkstra_calls"] += 1

    if math.isinf(dist.get(tgt, math.inf)):
        _finish(stats, t0, tracemalloc)
        return []

    # Collect the first path's edges.
    path0 = _reconstruct_sb(prev, src, tgt)
    if not path0:
        _finish(stats, t0, tracemalloc)
        return []

    # edge_pool accumulates ALL directed edges from ALL augmenting paths.
    edge_pool: dict[tuple[int, int], int] = defaultdict(int)
    for i in range(len(path0) - 1):
        edge_pool[(path0[i], path0[i + 1])] += 1

    used_edges: set[tuple[int, int]] = set(edge_pool.keys())

    # ── Iterations 1 … k-1: augmenting paths ──────────────────
    for _ in range(k - 1):
        stats["graph_transforms"] += 1
        trans_adj = _build_transformed_graph(
            all_ids, base_edges, used_edges, dist, stats
        )

        dist2, prev2 = dijkstra(trans_adj, src, all_ids)
        stats["dijkstra_calls"] += 1

        if math.isinf(dist2.get(tgt, math.inf)):
            break   # No more edge-disjoint paths.

        aug_path = _reconstruct_sb(prev2, src, tgt)
        if not aug_path:
            break

        # Add augmenting path edges to pool.
        for i in range(len(aug_path) - 1):
            edge_pool[(aug_path[i], aug_path[i + 1])] += 1

        # Update used_edges and dist for next iteration.
        used_edges = {e for e, cnt in edge_pool.items() if cnt > 0}
        dist = dist2  # Use reduced costs for next transform.

    # ── Extract simple paths from edge pool ───────────────────
    extracted = _extract_paths(all_ids, dict(edge_pool), src, tgt, k)

    _finish(stats, t0, tracemalloc)
    stats["paths_found"] = len(extracted)

    # Sort by actual cost.
    results = sorted(
        [(p, path_cost(g, p)) for p in extracted],
        key=lambda x: x[1],
    )

    return [(path, cost, stats) for path, cost in results]


def _finish(stats: dict, t0: float, tm) -> None:
    stats["elapsed_ms"]     = (time.perf_counter() - t0) * 1000
    _, peak = tm.get_traced_memory()
    stats["peak_memory_kb"] = peak // 1024
    tm.stop()


def _reconstruct_sb(prev: dict[int, int], src: int, tgt: int) -> list[int] | None:
    """Reconstruct path from predecessor map (same as graph.reconstruct)."""
    path: list[int] = []
    c: int | None = tgt
    seen: set[int] = set()
    while c is not None:
        if c in seen:
            return None   # Cycle detected — shouldn't happen with non-negative weights.
        seen.add(c)
        path.append(c)
        if c == src:
            break
        c = prev.get(c)
    path.reverse()
    return path if (path and path[0] == src) else None
