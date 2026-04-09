from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

Graph = Dict[int, List[int]]
Vertices = Set[int]
ReachabilityGraph = Dict[int, int]


def _compute_reachability_graph(live_graph: Graph) -> ReachabilityGraph:
    """Build transitive reachability bitmasks for all nodes in one sample."""
    reachability: ReachabilityGraph = {}

    for source in live_graph:
        visited_mask = 1 << source
        frontier = deque([source])

        while frontier:
            current_node = frontier.popleft()
            for neighbor in live_graph.get(current_node, []):
                neighbor_bit = 1 << neighbor
                if not (visited_mask & neighbor_bit):
                    visited_mask |= neighbor_bit
                    frontier.append(neighbor)

        reachability[source] = visited_mask

    return reachability


def generate_live_edge_subgraphs(
    graph: Graph,
    p: float,
    k: int,
    seed: int | None = None,
) -> List[ReachabilityGraph]:
    """Generate k sampled reachability maps for IC estimation."""
    if k <= 0:
        return []

    rng = random.Random(seed)
    rand = rng.random
    sampled_subgraphs: List[ReachabilityGraph] = []

    for _ in range(k):
        sampled_graph: Graph = {node: [] for node in graph}
        for node, neighbors in graph.items():
            live_neighbors = sampled_graph[node]
            for neighbor in neighbors:
                if rand() < p:
                    live_neighbors.append(neighbor)
        sampled_subgraphs.append(_compute_reachability_graph(sampled_graph))

    return sampled_subgraphs


def spread_on_live_edge_subgraph(graph: ReachabilityGraph, seed_set: Iterable[int]) -> int:
    """Count activated nodes by OR-union of all seeds' reachability bitmasks."""
    combined_mask = 0
    for seed in seed_set:
        if seed in graph:
            combined_mask |= graph[seed]
    return combined_mask.bit_count()


def average_spread_on_live_edge_subgraphs(
    sampled_subgraphs: List[ReachabilityGraph],
    seed_set: Iterable[int],
) -> float:
    """Average spread over sampled subgraphs (sum spread / k)."""
    if not sampled_subgraphs:
        return 0.0

    seeds = tuple(seed_set)
    total_spread = sum(spread_on_live_edge_subgraph(g, seeds) for g in sampled_subgraphs)

    return total_spread / len(sampled_subgraphs)


def load_graph(file_path: str | Path) -> Tuple[Graph, Vertices]:
    """Load a directed graph from an edge list file (u v per line)."""
    path = Path(file_path)
    graph: Graph = {}
    vertices: Vertices = set()

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            u, v = map(int, line.split())
            vertices.add(u)
            vertices.add(v)

            if u not in graph:
                graph[u] = []
            if v not in graph:
                graph[v] = []

            graph[u].append(v)

    print("-> Loaded graph successfully!")
    print(f"-> Total vertices |V|: {len(vertices)}")

    return graph, vertices


def monte_carlo_ic(
    graph: Graph,
    seed_set: Iterable[int],
    p: float = 0.01,
    r: int = 1000,
    seed: int | None = None,
) -> float:
    """Evaluate expected spread with the Independent Cascade model."""
    total_spread = 0
    max_node = max(graph.keys()) if graph else 0
    active_nodes = [False] * (max_node + 1)
    rng = random.Random(seed)
    rand = rng.random
    seeds = tuple(sorted(seed_set))

    for _ in range(r):
        new_active = deque()
        visited_in_this_run = []

        for source in seeds:
            if source <= max_node and not active_nodes[source]:
                active_nodes[source] = True
                new_active.append(source)
                visited_in_this_run.append(source)

        current_spread = len(visited_in_this_run)

        while new_active:
            current_node = new_active.popleft()
            if current_node not in graph:
                continue

            for neighbor in graph[current_node]:
                if not active_nodes[neighbor] and rand() < p:
                    active_nodes[neighbor] = True
                    new_active.append(neighbor)
                    visited_in_this_run.append(neighbor)
                    current_spread += 1

        total_spread += current_spread

        for node in visited_in_this_run:
            active_nodes[node] = False

    return total_spread / r if r > 0 else 0.0
