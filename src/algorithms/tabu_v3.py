from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from collections import deque
from typing import Deque, Dict, List, Set, Tuple

from src.algorithms.degree_heuristic import degree_heuristic
from src.algorithms.tabu_search import (
    _build_candidate_pool,
    _compute_solution_union_masks,
    _estimate_addition_profiles,
    _estimate_removal_profiles,
    _estimate_unreached_gains,
    _precompute_influence_scores,
    _rank_swap_moves,
    _select_removable_vertices,
)
from src.common.shared import (
    average_spread_on_live_edge_subgraphs,
    generate_live_edge_subgraphs,
    monte_carlo_ic,
)


@dataclass(frozen=True)
class _HistoryEntry:
    solution: Tuple[int, ...]
    estimate: float
    iter_id: int


@dataclass(frozen=True)
class _EliteEntry:
    solution: Tuple[int, ...]
    estimate: float
    mc_spread: float
    iter_id: int


def _solution_signature(solution: Set[int]) -> Tuple[int, ...]:
    return tuple(sorted(solution))


def _solution_distance(a: Set[int], b: Set[int]) -> int:
    return len(a.symmetric_difference(b))


def _append_history(
    history: Deque[_HistoryEntry],
    solution: Set[int],
    estimate: float,
    iter_id: int,
) -> None:
    history.append(_HistoryEntry(_solution_signature(solution), estimate, iter_id))


def _update_elite_archive(
    elite_archive: List[_EliteEntry],
    solution: Set[int],
    estimate: float,
    mc_spread: float,
    iter_id: int,
    elite_pool_size: int,
) -> None:
    if elite_pool_size <= 0:
        return

    signature = _solution_signature(solution)
    replaced = False
    for idx, elite in enumerate(elite_archive):
        if elite.solution == signature:
            if mc_spread > elite.mc_spread or (
                math.isclose(mc_spread, elite.mc_spread) and estimate > elite.estimate
            ):
                elite_archive[idx] = _EliteEntry(signature, estimate, mc_spread, iter_id)
            replaced = True
            break

    if not replaced:
        elite_archive.append(_EliteEntry(signature, estimate, mc_spread, iter_id))

    elite_archive.sort(key=lambda item: (-item.mc_spread, -item.estimate, item.iter_id))
    del elite_archive[elite_pool_size:]


def _choose_anchor_solution(
    current_solution: Set[int],
    history: Deque[_HistoryEntry],
    elite_archive: List[_EliteEntry],
    rng: random.Random,
) -> Tuple[Set[int] | None, str]:
    candidates: List[Tuple[int, float, int, Tuple[int, ...], str]] = []

    for item in elite_archive:
        candidate = set(item.solution)
        distance = _solution_distance(current_solution, candidate)
        if distance <= 0:
            continue
        candidates.append((distance, item.mc_spread, item.iter_id, item.solution, "elite"))

    for item in history:
        candidate = set(item.solution)
        distance = _solution_distance(current_solution, candidate)
        if distance <= 0:
            continue
        candidates.append((distance, item.estimate, item.iter_id, item.solution, "history"))

    if not candidates:
        return None, "none"

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    top_n = min(3, len(candidates))
    _, _, _, signature, source = rng.choice(candidates[:top_n])
    return set(signature), source


def _scale_candidate_mix(
    unreached_ratio: float,
    influence_ratio: float,
    random_ratio: float,
    diversify_random_ratio: float,
    diversify_mode: bool,
) -> Tuple[float, float, float]:
    if not diversify_mode:
        return unreached_ratio, influence_ratio, random_ratio

    random_ratio = min(1.0, max(0.0, diversify_random_ratio))
    remainder = max(0.0, 1.0 - random_ratio)
    base_sum = unreached_ratio + influence_ratio

    if base_sum > 0.0:
        scale = remainder / base_sum
        return unreached_ratio * scale, influence_ratio * scale, random_ratio

    return remainder, 0.0, random_ratio


def _apply_long_jump(
    anchor_solution: Set[int],
    jump_count: int,
    ordered_vertices: List[int],
    influence_score: Dict[int, float],
    out_degree: Dict[int, int],
    node_frequency: Dict[int, int],
    unreached_gain: Dict[int, float],
    rng: random.Random,
    diversify_random_ratio: float,
) -> Set[int]:
    if jump_count <= 0 or not anchor_solution:
        return set(anchor_solution)

    solution = set(anchor_solution)

    removable_ranked = sorted(
        solution,
        key=lambda node: (
            node_frequency.get(node, 0),
            influence_score.get(node, 0.0),
            out_degree.get(node, 0),
            node,
        ),
    )

    outside_solution = [node for node in ordered_vertices if node not in solution]
    if not outside_solution:
        return solution

    additions_ranked = sorted(
        outside_solution,
        key=lambda node: (
            -unreached_gain.get(node, 0.0),
            -influence_score.get(node, 0.0),
            -out_degree.get(node, 0),
            node_frequency.get(node, 0),
            node,
        ),
    )

    actual_swaps = min(jump_count, len(removable_ranked), len(additions_ranked))
    if actual_swaps <= 0:
        return solution

    remove_nodes = removable_ranked[:actual_swaps]
    for node in remove_nodes:
        solution.remove(node)

    random_add_count = int(round(actual_swaps * diversify_random_ratio))
    random_add_count = max(0, min(actual_swaps, random_add_count))
    greedy_add_count = actual_swaps - random_add_count

    add_nodes = additions_ranked[:greedy_add_count]

    remaining_pool = [node for node in additions_ranked if node not in add_nodes]
    if random_add_count > 0 and remaining_pool:
        if len(remaining_pool) <= random_add_count:
            add_nodes.extend(remaining_pool)
        else:
            add_nodes.extend(rng.sample(remaining_pool, random_add_count))

    for node in add_nodes:
        solution.add(node)

    while len(solution) < len(anchor_solution):
        fallback_pool = [node for node in ordered_vertices if node not in solution]
        if not fallback_pool:
            break
        solution.add(rng.choice(fallback_pool))

    while len(solution) > len(anchor_solution):
        solution.remove(rng.choice(tuple(solution)))

    return solution


def _decay_tabu_times(
    tabu_time: Dict[int, int],
    iter_id: int,
    decay: float,
) -> None:
    for node, expiry in list(tabu_time.items()):
        if expiry <= iter_id:
            continue
        remaining = expiry - iter_id
        new_remaining = int(round(remaining * decay))
        if new_remaining <= 0:
            tabu_time[node] = iter_id - 1
        else:
            tabu_time[node] = iter_id + new_remaining


def _should_restart(
    restart_policy: str,
    iter_id: int,
    no_improve_count: int,
    stagnation_trigger: int,
    restart_interval: int,
    cycle_hit: bool,
) -> Tuple[bool, str]:
    if restart_policy == "periodic":
        if restart_interval > 0 and iter_id > 0 and iter_id % restart_interval == 0:
            return True, "periodic"
        return False, "none"

    if restart_policy == "adaptive":
        if cycle_hit:
            return True, "cycle"
        if no_improve_count >= stagnation_trigger:
            return True, "stagnation"
        if restart_interval > 0 and iter_id > 0 and iter_id % restart_interval == 0:
            return True, "periodic"
        return False, "none"

    # Default policy: stagnation
    if no_improve_count >= stagnation_trigger:
        return True, "stagnation"
    return False, "none"


def tabu_search_v3(
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
    unreached_ratio: float = 0.80,
    influence_ratio: float = 0.00,
    random_ratio: float = 0.20,
    use_deep_backtracking: bool = True,
    backtrack_depth: int = 5,
    backtrack_trigger_ratio: float = 0.60,
    backtrack_tenure_decay: float = 0.70,
    elite_pool_size: int = 4,
    jump_ratio: float = 0.10,
    restart_policy: str = "adaptive",
    restart_interval: int = 50,
    diversify_random_ratio: float = 0.30,
    diversify_steps: int = 3,
    verbose: bool = True,
) -> Tuple[Set[int], dict]:
    """Run Tabu Search v3 with deep-backtracking diversification."""
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
    if backtrack_depth <= 0:
        raise ValueError("backtrack_depth must be positive")
    if restart_interval < 0:
        raise ValueError("restart_interval must be non-negative")
    if diversify_steps < 0:
        raise ValueError("diversify_steps must be non-negative")
    for ratio_name, ratio in (
        ("unreached_ratio", unreached_ratio),
        ("influence_ratio", influence_ratio),
        ("random_ratio", random_ratio),
        ("backtrack_trigger_ratio", backtrack_trigger_ratio),
        ("backtrack_tenure_decay", backtrack_tenure_decay),
        ("jump_ratio", jump_ratio),
        ("diversify_random_ratio", diversify_random_ratio),
    ):
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"{ratio_name} must be in [0, 1]")

    candidate_ratio_sum = unreached_ratio + influence_ratio + random_ratio
    if abs(candidate_ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            "candidate ratios must sum to 1.0: "
            "unreached_ratio + influence_ratio + random_ratio"
        )

    policy = restart_policy.strip().lower()
    if policy not in {"stagnation", "periodic", "adaptive"}:
        raise ValueError("restart_policy must be one of: stagnation, periodic, adaptive")

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
        "deep_backtracks": 0,
        "cycle_hits": 0,
        "periodic_restarts": 0,
        "diversify_iters": 0,
        "elite_archive_size_max": 0,
        "restart_policy": policy,
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
            "[TABU_V3] Phase 0 done: sample "
            f"{r} subgraphs in {timing['phase_sample_subgraphs']:.4f}s"
        )

    t_init_start = time.perf_counter()

    current_solution = set(degree_heuristic(graph, effective_k))
    best_global_solution: Set[int] = set(current_solution)

    ordered_vertices = sorted(vertices)
    out_degree = {node: len(graph.get(node, [])) for node in ordered_vertices}
    tabu_time = {node: 0 for node in ordered_vertices}
    node_frequency: Dict[int, int] = {node: 0 for node in ordered_vertices}
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

    history: Deque[_HistoryEntry] = deque(maxlen=max(2, backtrack_depth + 1))
    _append_history(history, current_solution, global_estimate_threshold, 0)

    elite_archive: List[_EliteEntry] = []
    _update_elite_archive(
        elite_archive,
        best_global_solution,
        global_estimate_threshold,
        global_best_spread,
        iter_id=0,
        elite_pool_size=elite_pool_size,
    )
    timing["elite_archive_size_max"] = len(elite_archive)

    seen_signatures: Dict[Tuple[int, ...], int] = {_solution_signature(current_solution): 0}

    timing["phase_init"] = time.perf_counter() - t_init_start

    if verbose:
        print(f"[TABU_V3] Phase 1 done: initialize in {timing['phase_init']:.4f}s")
        print(f"[TABU_V3] Start seed_count={len(best_global_solution)}")
        print(
            f"[TABU_V3] Start global_best (R={mc_validation_runs})={global_best_spread:.4f}"
        )
        print("[TABU_V3] Initial seed source: Degree heuristic")
        print(
            "[TABU_V3] Deep-backtracking: "
            f"enabled={use_deep_backtracking}, depth={backtrack_depth}, "
            f"trigger_ratio={backtrack_trigger_ratio:.2f}, jump_ratio={jump_ratio:.2f}"
        )
        print(
            "[TABU_V3] Candidate mix "
            f"unreached={unreached_ratio:.2f}/"
            f"influence={influence_ratio:.2f}/"
            f"random={random_ratio:.2f}"
        )

    t_search_start = time.perf_counter()
    iter_id = 1
    no_improve_count = 0
    sigma_current_solution = global_estimate_threshold
    diversify_iters_left = 0
    stagnation_trigger = max(1, int(math.ceil(backtrack_trigger_ratio * max_no_improve)))

    while iter_id <= max_iter and no_improve_count < max_no_improve:
        signature = _solution_signature(current_solution)
        last_seen = seen_signatures.get(signature)
        cycle_hit = (
            last_seen is not None
            and last_seen > 0
            and iter_id - last_seen <= max(3, backtrack_depth)
        )
        if cycle_hit:
            timing["cycle_hits"] += 1
        seen_signatures[signature] = iter_id

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

        should_restart, restart_reason = _should_restart(
            policy,
            iter_id,
            no_improve_count,
            stagnation_trigger,
            restart_interval,
            cycle_hit,
        )

        if use_deep_backtracking and should_restart:
            anchor_solution, anchor_source = _choose_anchor_solution(
                current_solution,
                history,
                elite_archive,
                rng,
            )
            if anchor_solution is not None:
                jump_count = max(1, int(round(jump_ratio * effective_k)))
                current_solution = _apply_long_jump(
                    anchor_solution,
                    jump_count,
                    ordered_vertices,
                    influence_score,
                    out_degree,
                    node_frequency,
                    unreached_gain,
                    rng,
                    diversify_random_ratio,
                )
                _decay_tabu_times(tabu_time, iter_id, backtrack_tenure_decay)
                sigma_current_solution = average_spread_on_live_edge_subgraphs(
                    sampled_subgraphs,
                    current_solution,
                )
                _append_history(history, current_solution, sigma_current_solution, iter_id)
                no_improve_count = max(0, no_improve_count // 2)
                diversify_iters_left = max(diversify_iters_left, diversify_steps)
                timing["deep_backtracks"] += 1
                if restart_reason == "periodic":
                    timing["periodic_restarts"] += 1

                if verbose:
                    print(
                        f"[TABU_V3] Iter={iter_id}: deep-backtrack ({restart_reason}), "
                        f"anchor={anchor_source}, jump={jump_count}, "
                        f"estimate={sigma_current_solution:.4f}"
                    )

        in_diversify_mode = diversify_iters_left > 0
        ratio_unreached, ratio_influence, ratio_random = _scale_candidate_mix(
            unreached_ratio,
            influence_ratio,
            random_ratio,
            diversify_random_ratio,
            in_diversify_mode,
        )
        if in_diversify_mode:
            diversify_iters_left -= 1
            timing["diversify_iters"] += 1

        candidate_pool, candidate_source = _build_candidate_pool(
            current_solution,
            out_degree,
            influence_score,
            c,
            rng,
            unreached_gain=unreached_gain,
            unreached_ratio=ratio_unreached,
            influence_ratio=ratio_influence,
            random_ratio=ratio_random,
        )

        if not candidate_pool:
            timing["stopped_no_valid_move"] = 1
            if verbose:
                print(f"[TABU_V3] Iter={iter_id}: empty candidate pool, stop early")
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
            limit=min(max(2, effective_k // 3), len(removal_loss)),
        )

        if not removable_vertices:
            timing["stopped_no_valid_move"] = 1
            if verbose:
                print(
                    f"[TABU_V3] Iter={iter_id}: no non-tabu removable vertex in S, stop early"
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

        ranked_moves = _rank_swap_moves(
            removable_vertices,
            candidate_pool,
            removal_loss,
            addition_gain,
        )

        if not ranked_moves:
            timing["stopped_no_valid_move"] = 1
            if verbose:
                print(f"[TABU_V3] Iter={iter_id}: no ranked swap move, stop early")
            break

        limit_eval = min(1200, len(ranked_moves))
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
            top_k = min(12, len(valid_moves))
            sigma_best_local, best_u, best_v = rng.choice(valid_moves[:top_k])

        if best_u is None or best_v is None:
            if use_deep_backtracking and len(history) > 1:
                no_improve_count += 1
                if verbose:
                    print(
                        f"[TABU_V3] Iter={iter_id}: no valid move, keep searching with backtracking"
                    )
                iter_id += 1
                continue

            timing["stopped_no_valid_move"] = 1
            if verbose:
                print(f"[TABU_V3] Iter={iter_id}: no valid move after tabu filter, stop early")
            break

        current_solution.remove(best_u)
        current_solution.add(best_v)
        sigma_current_solution = sigma_best_local

        tenure = max(1, int(round(rng.uniform(0.4, 0.8) * effective_k)))
        tabu_time[best_u] = iter_id + tenure
        tabu_time[best_v] = iter_id + tenure
        node_frequency[best_u] = node_frequency.get(best_u, 0) + 1
        node_frequency[best_v] = node_frequency.get(best_v, 0) + 1

        _append_history(history, current_solution, sigma_current_solution, iter_id)

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
                _update_elite_archive(
                    elite_archive,
                    best_global_solution,
                    global_estimate_threshold,
                    global_best_spread,
                    iter_id,
                    elite_pool_size,
                )
                timing["elite_archive_size_max"] = max(
                    timing["elite_archive_size_max"], len(elite_archive)
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
                f"[TABU_V3] Iter={iter_id}/{max_iter}, "
                f"local_best={sigma_best_local:.4f}, "
                f"global_best={global_best_spread:.4f}, "
                f"swap={best_u}->{best_v}, "
                f"v_source={selected_source}, "
                f"improved={improved}, "
                f"mc_verified={mc_verified}, "
                f"no_improve={no_improve_count}/{max_no_improve}, "
                f"diversify={int(in_diversify_mode)}, "
                f"S={sorted(current_solution)}"
            )

        timing["iterations_completed"] += 1
        iter_id += 1

    timing["phase_search"] = time.perf_counter() - t_search_start
    timing["global_best_spread"] = global_best_spread
    timing["final_no_improve_count"] = no_improve_count

    if verbose:
        print(f"[TABU_V3] Phase 2 done: search in {timing['phase_search']:.4f}s")
        print(
            "[TABU_V3] Detail: "
            f"iterations={timing['iterations_completed']}, "
            f"neighbor_evaluations={timing['neighbor_evaluations']}, "
            f"global_best={timing['global_best_spread']:.4f}, "
            f"deep_backtracks={timing['deep_backtracks']}, "
            f"cycle_hits={timing['cycle_hits']}"
        )

    return best_global_solution, timing
