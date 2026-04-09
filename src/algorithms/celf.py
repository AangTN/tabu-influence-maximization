from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from src.common.shared import (
    average_spread_on_live_edge_subgraphs,
    generate_live_edge_subgraphs,
)


@dataclass
class CELFNode:
    node: int
    marginal_gain: float
    iteration: int

    def __lt__(self, other: "CELFNode") -> bool:
        if self.marginal_gain != other.marginal_gain:
            return self.marginal_gain > other.marginal_gain
        if self.iteration != other.iteration:
            return self.iteration < other.iteration
        return self.node < other.node


def celf(
    graph: Dict[int, List[int]],
    vertices: Set[int],
    k: int,
    p: float = 0.3,
    num_subgraphs: int = 50,
    seed: int | None = None,
    subgraph_seed: int | None = None,
    verbose: bool = True,
    pre_sampled_subgraphs: List[Dict[int, int]] | None = None,
) -> Tuple[Set[int], dict]:
    """Run CELF using pre-sampled live-edge subgraphs."""
    if pre_sampled_subgraphs is None and num_subgraphs <= 0:
        raise ValueError("num_subgraphs must be a positive integer")
    if pre_sampled_subgraphs is not None and not pre_sampled_subgraphs:
        raise ValueError("pre_sampled_subgraphs must be non-empty when provided")

    q: List[CELFNode] = []
    selected: Set[int] = set()
    current_spread = 0.0

    timing = {
        "phase_sample_subgraphs": 0.0,
        "phase_init_queue": 0.0,
        "phase_k_loop": 0.0,
        "phase_recompute_only": 0.0,
        "stale_recompute_count": 0,
        "subgraph_source": "generated",
        "subgraph_count": 0,
    }

    resolved_subgraph_seed = seed if subgraph_seed is None else subgraph_seed

    if pre_sampled_subgraphs is None:
        t_sample_start = time.perf_counter()
        sampled_subgraphs = generate_live_edge_subgraphs(
            graph,
            p=p,
            k=num_subgraphs,
            seed=resolved_subgraph_seed,
        )
        timing["phase_sample_subgraphs"] = time.perf_counter() - t_sample_start
        timing["subgraph_source"] = "generated"

        if verbose:
            print(
                "[CELF] Phase 0 done: sample "
                f"{num_subgraphs} subgraphs in {timing['phase_sample_subgraphs']:.4f}s"
            )
    else:
        sampled_subgraphs = pre_sampled_subgraphs
        timing["phase_sample_subgraphs"] = 0.0
        timing["subgraph_source"] = "external"
        if verbose:
            print(
                "[CELF] Phase 0 skipped: reuse "
                f"{len(sampled_subgraphs)} pre-sampled subgraphs"
            )

    timing["subgraph_count"] = len(sampled_subgraphs)

    if not sampled_subgraphs:
        return set(), timing

    t_init_start = time.perf_counter()
    for u in sorted(vertices):
        marginal_gain = average_spread_on_live_edge_subgraphs(sampled_subgraphs, {u})
        q.append(CELFNode(node=u, marginal_gain=marginal_gain, iteration=0))
    heapq.heapify(q)
    timing["phase_init_queue"] = time.perf_counter() - t_init_start

    if verbose:
        print(f"[CELF] Phase 1 done: build queue in {timing['phase_init_queue']:.4f}s")

    t_loop_start = time.perf_counter()
    announced_k = 0

    while len(selected) < k and q:
        current_iteration = len(selected)
        current_k = current_iteration + 1

        if verbose and current_k != announced_k:
            print(f"[CELF] Running k={current_k}/{k}...")
            announced_k = current_k

        best_node = heapq.heappop(q)
        u = best_node.node

        if best_node.iteration == current_iteration:
            selected.add(u)
            current_spread += best_node.marginal_gain
            if verbose:
                print(
                    f"[CELF] k={len(selected)}/{k} selected node={u}, "
                    f"marginal_gain={best_node.marginal_gain:.4f}"
                )
        else:
            t_recompute_start = time.perf_counter()
            new_spread = average_spread_on_live_edge_subgraphs(
                sampled_subgraphs,
                selected | {u},
            )
            timing["phase_recompute_only"] += time.perf_counter() - t_recompute_start

            new_marginal_gain = max(0.0, new_spread - current_spread)
            heapq.heappush(
                q,
                CELFNode(node=u, marginal_gain=new_marginal_gain, iteration=current_iteration),
            )
            timing["stale_recompute_count"] += 1

    timing["phase_k_loop"] = time.perf_counter() - t_loop_start

    if verbose:
        print(f"[CELF] Phase 2 done: k-loop in {timing['phase_k_loop']:.4f}s")
        print(
            "[CELF] Recompute detail: "
            f"count={timing['stale_recompute_count']}, "
            f"time={timing['phase_recompute_only']:.4f}s"
        )

    return selected, timing
