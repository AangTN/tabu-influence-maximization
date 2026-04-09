from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Set

import networkx as nx
from networkx.algorithms.community import louvain_communities


def _build_undirected_graph(
    graph: Dict[int, List[int]],
    vertices: Set[int],
) -> nx.Graph:
    """Build an undirected NetworkX graph from a directed adjacency list."""
    undirected_graph = nx.Graph()
    undirected_graph.add_nodes_from(sorted(vertices))

    for u in sorted(graph):
        for v in sorted(graph.get(u, [])):
            undirected_graph.add_edge(u, v)

    return undirected_graph


def louvain_based_heuristic(
    graph: Dict[int, List[int]],
    k: int,
    vertices: Set[int],
    seed: int | None = None,
) -> List[int]:
    """Pick k seed nodes using Louvain communities and round-robin allocation."""
    if k <= 0 or not vertices:
        return []

    target_k = min(k, len(vertices))
    print(f"Finding top-{target_k} nodes with Louvain round-robin...")

    undirected_graph = _build_undirected_graph(graph, vertices)
    communities = louvain_communities(undirected_graph, seed=seed)

    node_to_community: Dict[int, int] = {}
    for community_id, members in enumerate(communities):
        for node in members:
            node_to_community[node] = community_id

    # Keep isolated or uncovered vertices in the partition map as singleton communities.
    next_community_id = len(communities)
    for node in sorted(vertices):
        if node not in node_to_community:
            node_to_community[node] = next_community_id
            next_community_id += 1

    out_degree = {node: len(graph.get(node, [])) for node in vertices}

    grouped_nodes: DefaultDict[int, List[int]] = defaultdict(list)
    for node in sorted(vertices):
        grouped_nodes[node_to_community[node]].append(node)

    for members in grouped_nodes.values():
        members.sort(key=lambda node: (-out_degree.get(node, 0), node))

    ordered_communities = sorted(
        grouped_nodes.values(),
        key=lambda members: (-len(members), members[0]),
    )

    seed_list: List[int] = []
    selected: Set[int] = set()
    pointers = [0 for _ in ordered_communities]

    while len(seed_list) < target_k:
        selected_in_round = False

        for idx, members in enumerate(ordered_communities):
            while pointers[idx] < len(members) and members[pointers[idx]] in selected:
                pointers[idx] += 1

            if pointers[idx] >= len(members):
                continue

            node = members[pointers[idx]]
            pointers[idx] += 1

            if node in selected:
                continue

            selected.add(node)
            seed_list.append(node)
            selected_in_round = True

            if len(seed_list) >= target_k:
                break

        if not selected_in_round:
            break

    if len(seed_list) < target_k:
        remaining_nodes = sorted(
            (node for node in vertices if node not in selected),
            key=lambda node: (-out_degree.get(node, 0), node),
        )
        for node in remaining_nodes:
            selected.add(node)
            seed_list.append(node)
            if len(seed_list) >= target_k:
                break

    return seed_list
