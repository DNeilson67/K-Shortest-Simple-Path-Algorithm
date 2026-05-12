"""
yen.py
------
Yen's K-Shortest Simple Paths algorithm (1971).

Reference
~~~~~~~~~
  Yen, J. Y. (1971). Finding the k shortest loopless paths in a network.
  Management Science, 17(11), 712–716.

Algorithm overview
~~~~~~~~~~~~~~~~~~
Yen's algorithm builds each new shortest simple path by "deviating" from
previously found paths at a chosen spur node:

  1. Find A[0] = shortest path s → t via Dijkstra.
  2. For i = 1 … k-1:
       For each spur node u along A[i-1] (all nodes except t):
         a. root_path = A[i-1] prefix up to u.
         b. Block edges leaving u that are used by prior paths sharing
            the same root (prevents duplicate candidates).
         c. Block all nodes already in the root (enforces simplicity).
         d. Dijkstra from u on the restricted graph → spur_path.
         e. candidate = root_path + spur_path  →  push to min-heap B.
       Pop the best candidate from B → A[i].

Complexity (theoretical)
~~~~~~~~~~~~~~~~~~~~~~~~
  Let n = |V|, m = |E|

  Time  : O(k · n · (m + n log n))
            k   outer iterations (one per new path)
          × n   spur nodes per iteration
          × O(m + n log n)  Dijkstra per spur node (binary heap)

  Space : O(k · n)  for storing k paths of length ≤ n
        + O(n + m)  for the graph and priority queue

Usage
~~~~~
  from graph import Graph
  from yen import yen_k_shortest

  result = yen_k_shortest(graph, source, target, k)
  # result: list of (path: list[int], cost: float, stats: dict)
"""

from __future__ import annotations
import heapq
import math
import time
import tracemalloc
from graph import Graph, dijkstra, reconstruct, path_cost


# ──────────────────────────────────────────────────────────────
# Complexity stats container
# ──────────────────────────────────────────────────────────────

def _empty_stats() -> dict:
    return {
        # Counters
        "dijkstra_calls":    0,   # number of Dijkstra invocations
        "heap_pushes":       0,   # candidates pushed into B
        "heap_pops":         0,   # candidates popped from B
        "paths_found":       0,   # confirmed paths returned
        # Complexity labels
        "time_complexity":   "O(k · n · (m + n log n))",
        "space_complexity":  "O(k · n)",
        # Measured
        "elapsed_ms":        0.0,
        "peak_memory_kb":    0,
    }


# ──────────────────────────────────────────────────────────────
# Main algorithm
# ──────────────────────────────────────────────────────────────

def yen_k_shortest(
    g: Graph,
    src: int,
    tgt: int,
    k: int,
) -> list[tuple[list[int], float, dict]]:
    """
    Find up to k shortest simple paths from src to tgt.

    Parameters
    ----------
    g   : Graph
    src : source node id
    tgt : target node id
    k   : number of paths requested

    Returns
    -------
    List of (path, cost, stats) tuples ordered by cost ascending.
    stats is a per-run dict with timing, memory, and operation counts.
    The same stats dict is shared/updated across all returned tuples.
    """
    stats = _empty_stats()
    tracemalloc.start()
    t0 = time.perf_counter()

    all_ids = g.all_ids()
    result: list[tuple[list[int], float]] = []

    # ── Step 1: shortest path ─────────────────────────────────
    adj0 = g.adj()
    _, prev0 = dijkstra(adj0, src, all_ids)
    stats["dijkstra_calls"] += 1

    first = reconstruct(prev0, src, tgt)
    if not first:
        _finish(stats, t0, tracemalloc)
        return []

    # A: confirmed paths;  B: candidate min-heap
    A: list[tuple[list[int], float]] = [(first, path_cost(g, first))]
    B: list[tuple[float, int, list[int]]] = []   # (cost, tie_break, path)
    seen_B: set[tuple[int, ...]] = set()
    tie = 0

    # ── Step 2: spur iterations ───────────────────────────────
    for _ in range(k - 1):
        prev_path, _ = A[-1]

        for si in range(len(prev_path) - 1):
            spur_node = prev_path[si]
            root      = prev_path[:si + 1]

            # Block edges that would duplicate an already-found prefix.
            blk_edges: set[tuple[int, int]] = set()
            for ap, _ in A:
                if (len(ap) > si
                        and ap[:si + 1] == root
                        and si + 1 < len(ap)):
                    blk_edges.add((ap[si], ap[si + 1]))

            # Block root nodes (except spur) to maintain simplicity.
            blk_nodes: set[int] = set(root[:-1])

            adj_r = g.adj(blk_nodes, blk_edges)
            ds, ps = dijkstra(adj_r, spur_node, all_ids)
            stats["dijkstra_calls"] += 1

            if math.isinf(ds[tgt]):
                continue

            spur_path = reconstruct(ps, spur_node, tgt)
            if not spur_path:
                continue

            candidate = root[:-1] + spur_path
            key = tuple(candidate)
            if key not in seen_B:
                seen_B.add(key)
                cost = path_cost(g, candidate)
                heapq.heappush(B, (cost, tie, candidate))
                stats["heap_pushes"] += 1
                tie += 1

        if not B:
            break   # No more simple paths exist.

        best_cost, _, best_path = heapq.heappop(B)
        stats["heap_pops"] += 1
        A.append((best_path, best_cost))

    _finish(stats, t0, tracemalloc)
    stats["paths_found"] = len(A)

    return [(path, cost, stats) for path, cost in A]


def _finish(stats: dict, t0: float, tm) -> None:
    stats["elapsed_ms"]     = (time.perf_counter() - t0) * 1000
    _, peak = tm.get_traced_memory()
    stats["peak_memory_kb"] = peak // 1024
    tm.stop()
