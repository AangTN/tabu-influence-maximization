from __future__ import annotations

import random
import time
from typing import Dict, List, Set, Tuple

from src.algorithms.advanced_neighborhoods import (
    compute_structural_signals,
    generate_advanced_neighborhood,
)
from src.algorithms.degree_heuristic import degree_heuristic
from src.algorithms.tabu_search import (
    _build_candidate_pool,
    _compute_solution_union_masks,
    _estimate_addition_profiles,
    _estimate_removal_profiles,
    _precompute_influence_scores,
    _rank_swap_moves,
    _select_removable_vertices,
)
from src.common.shared import (
    average_spread_on_live_edge_subgraphs,
    generate_live_edge_subgraphs,
    monte_carlo_ic,
)


def tabu_search_v2(
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
    neighborhood_strategy: str = "rcl_epsilon",
    seed: int | None = None,
    subgraph_seed: int | None = None,
    global_eval_runs: int = 1000,
    global_eval_seed: int | None = None,
    verbose: bool = True,
) -> Tuple[Set[int], dict]:
    """Run Tabu Search v2 with advanced neighborhood generation strategies."""
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

    strategy_key = neighborhood_strategy.strip().lower()
    supported = {
        "basic",
        "rcl_epsilon",
        "frequency_penalty",
        "core_periphery",
        "elite_relinking",
    }
    if strategy_key not in supported:
        raise ValueError(
            "Unknown strategy. Use one of: "
            "'basic', 'rcl_epsilon', 'frequency_penalty', 'core_periphery', 'elite_relinking'."
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
        "neighborhood_strategy": strategy_key,
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
            "[TABU_V2] Phase 0 done: sample "
            f"{r} subgraphs in {timing['phase_sample_subgraphs']:.4f}s"
        )

    t_init_start = time.perf_counter()

    current_solution = set(degree_heuristic(graph, effective_k))
    best_global_solution: Set[int] = set(current_solution)

    out_degree = {node: len(graph.get(node, [])) for node in vertices}
    tabu_time = {node: 0 for node in vertices}
    node_frequency: Dict[int, int] = {node: 0 for node in vertices}
    move_frequency: Dict[Tuple[int, int], int] = {}
    influence_score = _precompute_influence_scores(sampled_subgraphs, vertices)

    structural_core_number = None
    structural_bridge_score = None
    structural_out_degree = None
    structural_reverse_neighbors = None
    if strategy_key == "core_periphery":
        (
            structural_core_number,
            structural_bridge_score,
            structural_out_degree,
            structural_reverse_neighbors,
        ) = compute_structural_signals(graph)

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
        print(f"[TABU_V2] Phase 1 done: initialize in {timing['phase_init']:.4f}s")
        print(f"[TABU_V2] Start seed_count={len(best_global_solution)}")
        print(
            f"[TABU_V2] Start global_best (R={mc_validation_runs})={global_best_spread:.4f}"
        )
        print(f"[TABU_V2] Initial seed source: Degree heuristic (k={effective_k})")
        print("[TABU_V2] Tabu tenure: T = U(0.4, 0.8) * k")
        print("[TABU_V2] Neighbor mix fixed=80(influence)/20(random)")
        print(f"[TABU_V2] Neighborhood strategy: {strategy_key}")

    t_search_start = time.perf_counter()
    iter_id = 1
    no_improve_count = 0
    sigma_current_solution = global_estimate_threshold

    while iter_id <= max_iter and no_improve_count < max_no_improve:
        candidate_pool, candidate_source = _build_candidate_pool(
            current_solution,
            out_degree,
            influence_score,
            c,
            rng,
        )

        if not candidate_pool:
            timing["stopped_no_valid_move"] = 1
            if verbose:
                print(f"[TABU_V2] Iter={iter_id}: empty candidate pool, stop early")
            break

        union_masks, union_counts, sigma_current_solution = _compute_solution_union_masks(
            sampled_subgraphs,
            current_solution,
        )

        removal_loss, leave_one_out_unions = _estimate_removal_profiles(
            sampled_subgraphs,
            current_solution,
            sigma_current_solution,
            tabu_time,
            iter_id,
        )

        removable_vertices = _select_removable_vertices(
            removal_loss,
            limit=min(3, len(removal_loss)),
        )

        if not removable_vertices:
            timing["stopped_no_valid_move"] = 1
            if verbose:
                print(
                    f"[TABU_V2] Iter={iter_id}: no non-tabu removable vertex in S, stop early"
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

        if strategy_key == "basic":
            ranked_moves = _rank_swap_moves(
                removable_vertices,
                candidate_vertices,
                removal_loss,
                addition_gain,
            )
        else:
            ranked_moves = generate_advanced_neighborhood(
                strategy=strategy_key,
                graph=graph,
                current_solution=current_solution,
                removable_vertices=removable_vertices,
                candidate_vertices=candidate_vertices,
                removal_loss=removal_loss,
                addition_gain=addition_gain,
                tabu_time=tabu_time,
                iter_id=iter_id,
                rng=rng,
                stagnation_count=no_improve_count,
                stagnation_window=max_no_improve,
                node_frequency=node_frequency,
                move_frequency=move_frequency,
                elite_solution=best_global_solution,
                core_number=structural_core_number,
                bridge_score=structural_bridge_score,
                out_degree=structural_out_degree,
                reverse_neighbors=structural_reverse_neighbors,
            )

        if not ranked_moves:
            timing["stopped_no_valid_move"] = 1
            if verbose:
                print(f"[TABU_V2] Iter={iter_id}: no ranked swap move, stop early")
            break

        sigma_best_local = -1.0
        best_u: int | None = None
        best_v: int | None = None
        sample_count = len(sampled_subgraphs)

        for _, u, v in ranked_moves:
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
                if sigma_current > sigma_best_local:
                    sigma_best_local = sigma_current
                    best_u = u
                    best_v = v

        if best_u is None or best_v is None:
            timing["stopped_no_valid_move"] = 1
            if verbose:
                print(
                    f"[TABU_V2] Iter={iter_id}: no valid move after tabu filter, stop early"
                )
            break

        current_solution.remove(best_u)
        current_solution.add(best_v)
        sigma_current_solution = sigma_best_local
        tenure = max(1, int(round(rng.uniform(0.4, 0.8) * effective_k)))
        tabu_time[best_u] = iter_id + tenure
        tabu_time[best_v] = iter_id + tenure
        node_frequency[best_u] = node_frequency.get(best_u, 0) + 1
        node_frequency[best_v] = node_frequency.get(best_v, 0) + 1
        move_frequency[(best_u, best_v)] = move_frequency.get((best_u, best_v), 0) + 1

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
                f"[TABU_V2] Iter={iter_id}/{max_iter}, "
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
        print(f"[TABU_V2] Phase 2 done: search in {timing['phase_search']:.4f}s")
        print(
            "[TABU_V2] Detail: "
            f"iterations={timing['iterations_completed']}, "
            f"neighbor_evaluations={timing['neighbor_evaluations']}, "
            f"global_best={timing['global_best_spread']:.4f}"
        )

    return best_global_solution, timing
