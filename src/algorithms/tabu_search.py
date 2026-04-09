from __future__ import annotations

import random
import time
from typing import Dict, List, Set, Tuple

from src.algorithms.degree_heuristic import degree_heuristic
from src.common.shared import (
    average_spread_on_live_edge_subgraphs,
    generate_live_edge_subgraphs,
    monte_carlo_ic,
)

def _build_candidate_pool(
    current_solution: Set[int],
    out_degree: Dict[int, int],
    influence_score: Dict[int, float],
    c: int,
    rng: random.Random,
    unreached_gain: Dict[int, float] | None = None,
    unreached_ratio: float = 0.0,
    influence_ratio: float = 0.80,
    random_ratio: float = 0.20,
) -> Tuple[List[int], Dict[int, str]]:
    """Build candidate mix from unreached/influence/random sources."""
    if c <= 0:
        return [], {}

    use_unreached_tier = bool(unreached_gain) and unreached_ratio > 0.0
    if use_unreached_tier:
        unreached_quota = int(c * unreached_ratio)
        influence_quota = int(c * influence_ratio)
        random_quota = c - unreached_quota - influence_quota
    else:
        unreached_quota = 0
        influence_quota = int(c * 0.80)
        random_quota = c - influence_quota

    if random_quota < 0:
        raise ValueError("candidate quotas exceed pool size c")

    outside_solution = sorted(node for node in out_degree if node not in current_solution)
    if not outside_solution:
        return [], {}

    unreached_candidates: List[int] = []
    if use_unreached_tier and unreached_quota > 0:
        assert unreached_gain is not None
        unreached_candidates = sorted(
            outside_solution,
            key=lambda node: (
                -unreached_gain.get(node, 0.0),
                -influence_score.get(node, 0.0),
                -out_degree.get(node, 0),
                node,
            ),
        )[:unreached_quota]
    excluded = set(unreached_candidates)

    influence_candidates = sorted(
        (node for node in outside_solution if node not in excluded),
        key=lambda node: (
            -influence_score.get(node, 0.0),
            -out_degree.get(node, 0),
            node,
        ),
    )[:influence_quota]
    excluded.update(influence_candidates)

    random_pool = [node for node in outside_solution if node not in excluded]

    if random_quota <= 0:
        random_candidates: List[int] = []
    elif len(random_pool) <= random_quota:
        random_candidates = random_pool
    else:
        random_candidates = rng.sample(random_pool, random_quota)

    candidates = unreached_candidates + influence_candidates + random_candidates
    candidate_set = set(candidates)

    if len(candidates) < c:
        fallback_pool = [node for node in outside_solution if node not in candidate_set]
        if use_unreached_tier:
            assert unreached_gain is not None
            fallback_pool.sort(
                key=lambda node: (
                    -unreached_gain.get(node, 0.0),
                    -influence_score.get(node, 0.0),
                    -out_degree.get(node, 0),
                    node,
                )
            )
        else:
            fallback_pool.sort(
                key=lambda node: (
                    -influence_score.get(node, 0.0),
                    -out_degree.get(node, 0),
                    node,
                )
            )
        need = c - len(candidates)
        candidates.extend(fallback_pool[:need])

    source_by_node: Dict[int, str] = {}
    for node in unreached_candidates:
        source_by_node[node] = "unreached"
    for node in influence_candidates:
        source_by_node[node] = "influence"
    for node in random_candidates:
        source_by_node[node] = "random"
    for node in candidates:
        if node not in source_by_node:
            source_by_node[node] = "fallback"

    return candidates, source_by_node

def _precompute_influence_scores(
    sampled_subgraphs: List[Dict[int, int]],
    vertices: Set[int],
) -> Dict[int, float]:
    """Precompute singleton influence score over sampled subgraphs for each node."""
    if not sampled_subgraphs:
        return {node: 0.0 for node in sorted(vertices)}

    score_by_node: Dict[int, float] = {}
    sample_count = len(sampled_subgraphs)

    for node in sorted(vertices):
        total = 0
        for live_graph in sampled_subgraphs:
            total += live_graph.get(node, 0).bit_count()
        score_by_node[node] = total / sample_count

    return score_by_node


def _estimate_unreached_gains(
    sampled_subgraphs: List[Dict[int, int]],
    union_masks: List[int],
    candidate_vertices: List[int],
) -> Dict[int, float]:
    """Estimate average newly covered unreached nodes for each candidate."""
    if not sampled_subgraphs or not union_masks or not candidate_vertices:
        return {}

    sample_count = len(sampled_subgraphs)
    unreached_gain_sums = {node: 0 for node in candidate_vertices}

    for idx, live_graph in enumerate(sampled_subgraphs):
        unreached_mask = ~union_masks[idx]
        for node in candidate_vertices:
            node_mask = live_graph.get(node, 0)
            unreached_gain_sums[node] += (node_mask & unreached_mask).bit_count()

    return {
        node: unreached_gain_sums[node] / sample_count
        for node in candidate_vertices
    }


def _compute_solution_union_masks(
    sampled_subgraphs: List[Dict[int, int]],
    current_solution: Set[int],
) -> Tuple[List[int], List[int], float]:
    """Compute per-sample OR-union masks and current average spread for S."""
    if not sampled_subgraphs:
        return [], [], 0.0

    seeds = tuple(current_solution)
    union_masks: List[int] = []
    union_counts: List[int] = []
    total_count = 0

    for live_graph in sampled_subgraphs:
        union_mask = 0
        for seed in seeds:
            union_mask |= live_graph.get(seed, 0)

        count = union_mask.bit_count()
        union_masks.append(union_mask)
        union_counts.append(count)
        total_count += count

    return union_masks, union_counts, total_count / len(sampled_subgraphs)


def _estimate_removal_losses(
    sampled_subgraphs: List[Dict[int, int]],
    current_solution: Set[int],
    current_avg_spread: float,
    tabu_time: Dict[int, int],
    iter_id: int,
) -> Dict[int, float]:
    """Estimate removal impact loss(u)=f(S)-f(S\\{u}) for each non-tabu u in S."""
    loss_by_vertex, _ = _estimate_removal_profiles(
        sampled_subgraphs,
        current_solution,
        current_avg_spread,
        tabu_time,
        iter_id,
    )
    return loss_by_vertex


def _estimate_removal_profiles(
    sampled_subgraphs: List[Dict[int, int]],
    current_solution: Set[int],
    current_avg_spread: float,
    tabu_time: Dict[int, int],
    iter_id: int,
) -> Tuple[Dict[int, float], Dict[int, List[int]]]:
    """Estimate loss(u) and cache exact per-sample unions for S\\{u}."""
    if not sampled_subgraphs:
        return {}, {}

    leave_one_out_unions = _build_leave_one_out_unions(
        sampled_subgraphs,
        current_solution,
        tabu_time,
        iter_id,
    )
    if not leave_one_out_unions:
        return {}, {}

    sample_count = len(sampled_subgraphs)
    loss_by_vertex: Dict[int, float] = {}
    for removed, union_masks in leave_one_out_unions.items():
        spread_without_removed = sum(mask.bit_count() for mask in union_masks) / sample_count
        loss_by_vertex[removed] = current_avg_spread - spread_without_removed

    return loss_by_vertex, leave_one_out_unions


def _build_leave_one_out_unions(
    sampled_subgraphs: List[Dict[int, int]],
    current_solution: Set[int],
    tabu_time: Dict[int, int],
    iter_id: int,
) -> Dict[int, List[int]]:
    """Build exact per-sample union mask of S\\{u} for each non-tabu u in S."""
    if not sampled_subgraphs or not current_solution:
        return {}

    seeds = tuple(current_solution)
    free_by_index: Dict[int, int] = {
        index: seed
        for index, seed in enumerate(seeds)
        if iter_id > tabu_time[seed]
    }
    if not free_by_index:
        return {}

    sample_count = len(sampled_subgraphs)
    seed_count = len(seeds)
    leave_one_out_unions: Dict[int, List[int]] = {
        seed: [0] * sample_count for seed in free_by_index.values()
    }

    for sample_idx, live_graph in enumerate(sampled_subgraphs):
        seed_masks = [live_graph.get(seed, 0) for seed in seeds]
        prefix_or = [0] * (seed_count + 1)

        for idx, mask in enumerate(seed_masks):
            prefix_or[idx + 1] = prefix_or[idx] | mask

        suffix_or = 0
        for idx in range(seed_count - 1, -1, -1):
            removed = free_by_index.get(idx)
            if removed is not None:
                leave_one_out_unions[removed][sample_idx] = prefix_or[idx] | suffix_or
            suffix_or |= seed_masks[idx]

    return leave_one_out_unions


def _estimate_addition_profiles(
    sampled_subgraphs: List[Dict[int, int]],
    union_masks: List[int],
    union_counts: List[int],
    candidate_pool: List[int],
) -> Tuple[Dict[int, float], Dict[int, List[int]]]:
    """Estimate gain(v) and cache per-sample masks of each candidate v."""
    if not sampled_subgraphs or not candidate_pool:
        return {}, {}

    sample_count = len(sampled_subgraphs)
    gain_sums = {node: 0 for node in candidate_pool}
    candidate_masks = {node: [0] * sample_count for node in candidate_pool}

    for idx, live_graph in enumerate(sampled_subgraphs):
        union_mask = union_masks[idx]
        union_count = union_counts[idx]
        for node in candidate_pool:
            node_mask = live_graph.get(node, 0)
            candidate_masks[node][idx] = node_mask
            gain_sums[node] += (union_mask | node_mask).bit_count() - union_count

    gain_by_vertex = {
        node: gain_sums[node] / sample_count for node in candidate_pool
    }

    return gain_by_vertex, candidate_masks


def _select_removable_vertices(
    removal_loss: Dict[int, float],
    limit: int,
) -> List[int]:
    """Pick low-loss vertices first to avoid removing highly critical seeds."""
    ranked = sorted(removal_loss, key=lambda node: (removal_loss[node], node))
    return ranked[:limit]


def _estimate_addition_gains(
    sampled_subgraphs: List[Dict[int, int]],
    union_masks: List[int],
    union_counts: List[int],
    candidate_pool: List[int],
) -> Dict[int, float]:
    """Estimate gain(v)=f(S U {v})-f(S) for each candidate v in V\\S."""
    gain_by_vertex, _ = _estimate_addition_profiles(
        sampled_subgraphs,
        union_masks,
        union_counts,
        candidate_pool,
    )
    return gain_by_vertex


def _rank_swap_moves(
    removable_vertices: List[int],
    candidate_vertices: List[int],
    removal_loss: Dict[int, float],
    addition_gain: Dict[int, float],
) -> List[Tuple[float, int, int]]:
    """Rank swaps by approx delta(v,u)=gain(v)-loss(u), descending."""
    ranked_moves: List[Tuple[float, int, int]] = []
    for u in removable_vertices:
        loss_u = removal_loss.get(u, 0.0)
        for v in candidate_vertices:
            score = addition_gain.get(v, 0.0) - loss_u
            ranked_moves.append((score, u, v))

    ranked_moves.sort(key=lambda item: (-item[0], item[1], item[2]))
    return ranked_moves


def tabu_search(
    graph: Dict[int, List[int]],
    vertices: Set[int],
    k: int,
    p: float = 0.3,
    c: int = 100,
    r: int = 100,
    t_min: int = 3,
    t_max: int = 7,
    max_iter: int = 100,
    max_no_improve: int = 20,
    seed: int | None = None,
    subgraph_seed: int | None = None,
    global_eval_runs: int = 1000,
    global_eval_seed: int | None = None,
    unreached_ratio: float = 0.40,
    influence_ratio: float = 0.40,
    random_ratio: float = 0.10,
    verbose: bool = True,
) -> Tuple[Set[int], dict]:
    """Run Tabu Search for influence maximization on live-edge subgraphs."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if c <= 0:
        raise ValueError("c must be a positive integer")
    if r <= 0:
        raise ValueError("r must be a positive integer")
    if t_min < 0:
        raise ValueError("t_min must be non-negative")
    if t_max < 0:
        raise ValueError("t_max must be non-negative")
    if t_min > t_max:
        raise ValueError("t_min must be less than or equal to t_max")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if max_no_improve <= 0:
        raise ValueError("max_no_improve must be a positive integer")
    for ratio_name, ratio in (
        ("unreached_ratio", unreached_ratio),
        ("influence_ratio", influence_ratio),
        ("random_ratio", random_ratio),
    ):
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"{ratio_name} must be in [0, 1]")

    candidate_ratio_sum = unreached_ratio + influence_ratio + random_ratio
    if abs(candidate_ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            "candidate ratios must sum to 1.0: "
            "unreached_ratio + influence_ratio + random_ratio"
        )

    timing = {
        "phase_sample_subgraphs": 0.0,
        "phase_init": 0.0,
        "phase_search": 0.0,
        "iterations_completed": 0,
        "neighbor_evaluations": 0,
        "global_mc_validations": 0,
        "global_best_spread": 0.0,
        "final_no_improve_count": 0,
        "stopped_no_valid_move": 0,
    }

    if not vertices or k == 0:
        return set(), timing

    effective_k = min(k, len(vertices))
    rng = random.Random(seed)
    resolved_subgraph_seed = seed if subgraph_seed is None else subgraph_seed

    t_sample_start = time.perf_counter()
    sampled_subgraphs = generate_live_edge_subgraphs(
        graph,
        p=p,
        k=r,
        seed=resolved_subgraph_seed,
    )
    timing["phase_sample_subgraphs"] = time.perf_counter() - t_sample_start

    if verbose:
        print(
            "[TABU] Phase 0 done: sample "
            f"{r} subgraphs in {timing['phase_sample_subgraphs']:.4f}s"
        )

    t_init_start = time.perf_counter()

    current_solution = set(degree_heuristic(graph, effective_k))
    best_global_solution: Set[int] = set(current_solution)

    ordered_vertices = sorted(vertices)
    out_degree = {node: len(graph.get(node, [])) for node in ordered_vertices}
    tabu_time = {node: 0 for node in ordered_vertices}
    influence_score = _precompute_influence_scores(sampled_subgraphs, vertices)

    global_estimate_threshold = average_spread_on_live_edge_subgraphs(
        sampled_subgraphs,
        best_global_solution,
    )
    mc_validation_runs = global_eval_runs
    mc_validation_seed = global_eval_seed
    global_best_spread = monte_carlo_ic(
        graph,
        best_global_solution,
        p=p,
        r=mc_validation_runs,
        seed=mc_validation_seed,
    )

    timing["phase_init"] = time.perf_counter() - t_init_start

    if verbose:
        print(f"[TABU] Phase 1 done: initialize in {timing['phase_init']:.4f}s")
        print(f"[TABU] Start seed_count={len(best_global_solution)}")
        print(
            f"[TABU] Start global_best (R={mc_validation_runs})={global_best_spread:.4f}"
        )
        print(f"[TABU] Initial seed source: Degree heuristic (k={effective_k})")
        print("[TABU] Tabu tenure: T = U(0.5, 0.9) * k")
        print(
            "[TABU] Neighbor mix "
            f"unreached={unreached_ratio:.2f}/"
            f"influence={influence_ratio:.2f}/"
            f"random={random_ratio:.2f}"
        )

    t_search_start = time.perf_counter()
    iter_id = 1
    no_improve_count = 0
    sigma_current_solution = global_estimate_threshold

    while iter_id <= max_iter and no_improve_count < max_no_improve:
        union_masks, union_counts, sigma_current_solution = _compute_solution_union_masks(
            sampled_subgraphs,
            current_solution,
        )

        outside_solution = [
            node for node in ordered_vertices if node not in current_solution
        ]
        unreached_gain = _estimate_unreached_gains(
            sampled_subgraphs,
            union_masks,
            outside_solution,
        )

        candidate_pool, candidate_source = _build_candidate_pool(
            current_solution,
            out_degree,
            influence_score,
            c,
            rng,
            unreached_gain=unreached_gain,
            unreached_ratio=unreached_ratio,
            influence_ratio=influence_ratio,
            random_ratio=random_ratio,
        )

        if not candidate_pool:
            timing["stopped_no_valid_move"] = 1
            if verbose:
                print(f"[TABU] Iter={iter_id}: empty candidate pool, stop early")
            break

        removal_loss, leave_one_out_unions = _estimate_removal_profiles(
            sampled_subgraphs,
            current_solution,
            sigma_current_solution,
            tabu_time,
            iter_id,
        )

        removable_vertices = _select_removable_vertices(
            removal_loss,
            limit=min(max(1, k // 2), len(removal_loss)),
        )

        if not removable_vertices:
            timing["stopped_no_valid_move"] = 1
            if verbose:
                print(
                    f"[TABU] Iter={iter_id}: no non-tabu removable vertex in S, stop early"
                )
            break

        addition_gain, candidate_masks = _estimate_addition_profiles(
            sampled_subgraphs,
            union_masks,
            union_counts,
            candidate_pool,
        )

        if not addition_gain:
            timing["stopped_no_valid_move"] = 1
            break

        candidate_vertices = candidate_pool

        ranked_moves = _rank_swap_moves(
            removable_vertices,
            candidate_vertices,
            removal_loss,
            addition_gain,
        )

        if not ranked_moves:
            timing["stopped_no_valid_move"] = 1
            if verbose:
                print(f"[TABU] Iter={iter_id}: no ranked swap move, stop early")
            break

        limit_eval = min(1000, len(ranked_moves))
        top_ranked_moves = ranked_moves[:limit_eval]

        sigma_best_local = -1.0
        best_u: int | None = None
        best_v: int | None = None
        sample_count = len(sampled_subgraphs)
        valid_moves: List[Tuple[float, int, int]] = []

        for _, u, v in top_ranked_moves:
            removed_unions = leave_one_out_unions.get(u)
            added_masks = candidate_masks.get(v)
            if removed_unions is None or added_masks is None:
                continue

            spread_sum = 0
            for idx in range(sample_count):
                spread_sum += (removed_unions[idx] | added_masks[idx]).bit_count()
            sigma_current = spread_sum / sample_count
            timing["neighbor_evaluations"] += 1

            is_tabu = iter_id <= tabu_time[v]

            if (not is_tabu) or (sigma_current > global_estimate_threshold):
                valid_moves.append((sigma_current, u, v))

        if valid_moves:
            valid_moves.sort(key=lambda item: (-item[0], item[1], item[2]))
            top_k = min(10, len(valid_moves))
            sigma_best_local, best_u, best_v = rng.choice(valid_moves[:top_k])

        if best_u is None or best_v is None:
            timing["stopped_no_valid_move"] = 1
            if verbose:
                print(f"[TABU] Iter={iter_id}: no valid move after tabu filter, stop early")
            break

        current_solution.remove(best_u)
        current_solution.add(best_v)
        sigma_current_solution = sigma_best_local
        tenure = max(1, int(round(rng.uniform(0.4, 0.8) * effective_k)))
        tabu_time[best_u] = iter_id + tenure
        tabu_time[best_v] = iter_id + tenure

        if sigma_best_local > global_estimate_threshold:
            timing["global_mc_validations"] += 1
            candidate_mc_spread = monte_carlo_ic(
                graph,
                current_solution,
                p=p,
                r=mc_validation_runs,
                seed=mc_validation_seed,
            )

            if candidate_mc_spread > global_best_spread:
                best_global_solution = set(current_solution)
                global_best_spread = candidate_mc_spread
                global_estimate_threshold = average_spread_on_live_edge_subgraphs(
                    sampled_subgraphs,
                    best_global_solution,
                )
                no_improve_count = 0
                improved = "yes"
                mc_verified = "yes"
            else:
                no_improve_count += 1
                improved = "no"
                mc_verified = "no"
        else:
            no_improve_count += 1
            improved = "no"
            mc_verified = "n/a"

        if verbose:
            selected_source = candidate_source.get(best_v, "unknown")
            print(
                f"[TABU] Iter={iter_id}/{max_iter}, "
                f"local_best={sigma_best_local:.4f}, "
                f"global_best={global_best_spread:.4f}, "
                f"swap={best_u}->{best_v}, "
                f"v_source={selected_source}, "
                f"improved={improved}, "
                f"mc_verified={mc_verified}, "
                f"screened={len(ranked_moves)}/{len(ranked_moves)}, "
                f"no_improve={no_improve_count}/{max_no_improve}, "
                f"S={sorted(current_solution)}"
            )

        timing["iterations_completed"] += 1
        iter_id += 1

    timing["phase_search"] = time.perf_counter() - t_search_start
    timing["global_best_spread"] = global_best_spread
    timing["final_no_improve_count"] = no_improve_count

    if verbose:
        print(f"[TABU] Phase 2 done: search in {timing['phase_search']:.4f}s")
        print(
            "[TABU] Detail: "
            f"iterations={timing['iterations_completed']}, "
            f"neighbor_evaluations={timing['neighbor_evaluations']}, "
            f"global_best={timing['global_best_spread']:.4f}"
        )

    return best_global_solution, timing