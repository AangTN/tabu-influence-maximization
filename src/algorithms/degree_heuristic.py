from __future__ import annotations

from typing import Dict, List


def degree_heuristic(graph: Dict[int, List[int]], k: int) -> List[int]:
    """Pick k nodes with highest out-degree."""
    print(f"Finding top-{k} nodes by out-degree...")

    out_degrees = {node: len(graph[node]) for node in graph}
    sorted_nodes = sorted(out_degrees, key=lambda x: out_degrees[x], reverse=True)
    return sorted_nodes[:k]
