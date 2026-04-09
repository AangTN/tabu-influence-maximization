from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import networkx as nx

Graph = Dict[int, List[int]]
ReachabilityGraph = Dict[int, int]
Move = Tuple[float, int, int]  # (score, u_remove, v_add)


def _validate_probability(value: float, name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _build_feasible_moves(
    current_solution: Set[int],
    removable_vertices: Sequence[int],
    candidate_vertices: Sequence[int],
) -> List[Tuple[int, int]]:
    feasible: List[Tuple[int, int]] = []
    for u in removable_vertices:
        if u not in current_solution:
            continue
        for v in candidate_vertices:
            if v in current_solution or v == u:
                continue
            feasible.append((u, v))
    return feasible


def _base_swap_score(
    u: int,
    v: int,
    removal_loss: Mapping[int, float],
    addition_gain: Mapping[int, float],
) -> float:
    return addition_gain.get(v, 0.0) - removal_loss.get(u, 0.0)


def _is_tabu_addition(
    v: int,
    tabu_time: Mapping[int, int],
    iter_id: int,
) -> bool:
    return iter_id <= tabu_time.get(v, 0)


def _sort_moves_desc(moves: Iterable[Move]) -> List[Move]:
    ranked = list(moves)
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    return ranked


def _resolve_adaptive_epsilon(
    stagnation_count: int,
    stagnation_window: int,
    epsilon_min: float,
    epsilon_max: float,
) -> float:
    _validate_probability(epsilon_min, "epsilon_min")
    _validate_probability(epsilon_max, "epsilon_max")
    if epsilon_min > epsilon_max:
        raise ValueError("epsilon_min must be <= epsilon_max")
    if stagnation_window <= 0:
        return epsilon_max

    ratio = min(1.0, max(0.0, stagnation_count / stagnation_window))
    return epsilon_min + (epsilon_max - epsilon_min) * ratio


def neighborhood_rcl_epsilon_greedy(
    current_solution: Set[int],
    removable_vertices: Sequence[int],
    candidate_vertices: Sequence[int],
    removal_loss: Mapping[int, float],
    addition_gain: Mapping[int, float],
    tabu_time: Mapping[int, int],
    iter_id: int,
    rng: random.Random,
    *,
    rcl_size: int = 20,
    epsilon_min: float = 0.05,
    epsilon_max: float = 0.35,
    stagnation_count: int = 0,
    stagnation_window: int = 20,
    aspiration_baseline: float | None = None,
) -> List[Move]:
    """RCL + adaptive epsilon-greedy ranking.

    Logic:
    - Rank feasible swaps by base delta.
    - Keep top-RCL as promising neighborhood.
    - Select one move by epsilon-greedy where epsilon increases with stagnation.

    Trade-off:
    - Improves escape from deep local optima.
    - Excessive epsilon can slow intensification.
    """
    if rcl_size <= 0:
        raise ValueError("rcl_size must be positive")

    feasible = _build_feasible_moves(current_solution, removable_vertices, candidate_vertices)
    if not feasible:
        return []

    scored: List[Move] = []
    for u, v in feasible:
        base_score = _base_swap_score(u, v, removal_loss, addition_gain)
        tabu = _is_tabu_addition(v, tabu_time, iter_id)
        aspirated = aspiration_baseline is not None and base_score > aspiration_baseline
        if tabu and not aspirated:
            continue
        scored.append((base_score, u, v))

    if not scored:
        return []

    ranked = _sort_moves_desc(scored)
    rcl = ranked[: min(rcl_size, len(ranked))]

    epsilon = _resolve_adaptive_epsilon(
        stagnation_count=stagnation_count,
        stagnation_window=stagnation_window,
        epsilon_min=epsilon_min,
        epsilon_max=epsilon_max,
    )

    if len(rcl) == 1:
        selected = rcl[0]
    elif rng.random() < epsilon:
        selected = rcl[rng.randrange(len(rcl))]
    else:
        selected = rcl[0]

    # Put the selected move first so external loops can pick index 0 directly.
    ordered = [selected]
    ordered.extend(move for move in ranked if move != selected)
    return ordered


def neighborhood_frequency_penalized(
    current_solution: Set[int],
    removable_vertices: Sequence[int],
    candidate_vertices: Sequence[int],
    removal_loss: Mapping[int, float],
    addition_gain: Mapping[int, float],
    tabu_time: Mapping[int, int],
    iter_id: int,
    *,
    node_frequency: Mapping[int, int] | None = None,
    move_frequency: Mapping[Tuple[int, int], int] | None = None,
    node_penalty_weight: float = 0.10,
    move_penalty_weight: float = 0.20,
    aspiration_baseline: float | None = None,
) -> List[Move]:
    """Frequency memory strategy with long-term diversification penalties.

    Logic:
    - Penalize nodes and swaps that appear too often in historical moves.
    - Keep high base-quality moves but discount repetitive patterns.

    Trade-off:
    - Reduces cycling and promotes coverage of unseen regions.
    - Over-penalization can suppress strong recurrent motifs.
    """
    if node_penalty_weight < 0.0 or move_penalty_weight < 0.0:
        raise ValueError("Penalty weights must be non-negative")

    node_frequency = node_frequency or {}
    move_frequency = move_frequency or {}

    feasible = _build_feasible_moves(current_solution, removable_vertices, candidate_vertices)
    if not feasible:
        return []

    scored: List[Move] = []
    for u, v in feasible:
        base_score = _base_swap_score(u, v, removal_loss, addition_gain)
        tabu = _is_tabu_addition(v, tabu_time, iter_id)
        aspirated = aspiration_baseline is not None and base_score > aspiration_baseline
        if tabu and not aspirated:
            continue

        node_penalty = node_penalty_weight * (node_frequency.get(u, 0) + node_frequency.get(v, 0))
        swap_penalty = move_penalty_weight * move_frequency.get((u, v), 0)
        score = base_score - node_penalty - swap_penalty
        scored.append((score, u, v))

    return _sort_moves_desc(scored)


def compute_structural_signals(
    graph: Graph,
) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, int], Dict[int, Set[int]]]:
    """Compute k-core, bridge score, out-degree, and reverse-neighbor index.

    Returns:
    - core_number: undirected k-core index per node.
    - bridge_score: fraction of neighbors with different core index.
    - out_degree: out-degree per node in directed graph.
    - reverse_neighbors: incoming-neighbor sets to support fast frontier affinity.
    """
    nodes = sorted(graph)
    undirected = nx.Graph()
    undirected.add_nodes_from(nodes)

    reverse_neighbors: Dict[int, Set[int]] = {node: set() for node in nodes}
    out_degree = {node: len(graph.get(node, [])) for node in nodes}

    for u in nodes:
        for v in graph.get(u, []):
            undirected.add_edge(u, v)
            if v not in reverse_neighbors:
                reverse_neighbors[v] = set()
            reverse_neighbors[v].add(u)
            if u not in reverse_neighbors:
                reverse_neighbors[u] = set()

    if undirected.number_of_nodes() == 0:
        return {}, {}, {}, reverse_neighbors

    core_number = nx.core_number(undirected)

    bridge_score: Dict[int, float] = {}
    for node in undirected.nodes:
        neighbors = list(undirected.neighbors(node))
        if not neighbors:
            bridge_score[node] = 0.0
            continue

        node_core = core_number.get(node, 0)
        cross_core = sum(1 for nbr in neighbors if core_number.get(nbr, 0) != node_core)
        bridge_score[node] = cross_core / len(neighbors)

    return core_number, bridge_score, out_degree, reverse_neighbors


def neighborhood_structural_core_periphery(
    graph: Graph,
    current_solution: Set[int],
    removable_vertices: Sequence[int],
    candidate_vertices: Sequence[int],
    removal_loss: Mapping[int, float],
    addition_gain: Mapping[int, float],
    tabu_time: Mapping[int, int],
    iter_id: int,
    *,
    core_number: Mapping[int, int] | None = None,
    bridge_score: Mapping[int, float] | None = None,
    out_degree: Mapping[int, int] | None = None,
    reverse_neighbors: Mapping[int, Set[int]] | None = None,
    remove_limit: int = 6,
    add_limit: int = 30,
    core_weight: float = 0.35,
    bridge_weight: float = 0.50,
    frontier_weight: float = 0.15,
    aspiration_baseline: float | None = None,
) -> List[Move]:
    """Core-periphery neighborhood for structural diversification.

    Logic:
    - Prefer removing saturated core seeds (high core index).
    - Prefer adding periphery-bridge nodes with frontier affinity to current seeds.
    - Combine structural bonus with base gain-loss score.

    Trade-off:
    - Escapes topology-induced traps missed by pure marginal gain.
    - Adds structural preprocessing overhead.
    """
    if remove_limit <= 0 or add_limit <= 0:
        raise ValueError("remove_limit and add_limit must be positive")

    if core_number is None or bridge_score is None or out_degree is None or reverse_neighbors is None:
        core_number, bridge_score, out_degree, reverse_neighbors = compute_structural_signals(graph)

    if not core_number:
        return neighborhood_rcl_epsilon_greedy(
            current_solution=current_solution,
            removable_vertices=removable_vertices,
            candidate_vertices=candidate_vertices,
            removal_loss=removal_loss,
            addition_gain=addition_gain,
            tabu_time=tabu_time,
            iter_id=iter_id,
            rng=random.Random(0),
            rcl_size=20,
            epsilon_min=0.0,
            epsilon_max=0.0,
            stagnation_count=0,
            stagnation_window=1,
            aspiration_baseline=aspiration_baseline,
        )

    max_core = max(core_number.values()) if core_number else 1
    max_degree = max(out_degree.values()) if out_degree else 1

    def core_norm(node: int) -> float:
        denom = max(1, max_core)
        return core_number.get(node, 0) / denom

    def degree_norm(node: int) -> float:
        denom = max(1, max_degree)
        return out_degree.get(node, 0) / denom

    def frontier_affinity(node: int) -> float:
        out_links = sum(1 for nbr in graph.get(node, []) if nbr in current_solution)
        in_links = sum(1 for nbr in reverse_neighbors.get(node, set()) if nbr in current_solution)
        total = out_links + in_links
        denom = max(1, len(current_solution))
        return total / denom

    removable_ranked = sorted(
        [u for u in removable_vertices if u in current_solution],
        key=lambda u: (-core_norm(u), removal_loss.get(u, 0.0), u),
    )[:remove_limit]

    candidate_ranked = sorted(
        [v for v in candidate_vertices if v not in current_solution],
        key=lambda v: (
            -(
                bridge_weight * bridge_score.get(v, 0.0) * (1.0 - core_norm(v))
                + frontier_weight * frontier_affinity(v)
                + 0.20 * degree_norm(v)
            ),
            v,
        ),
    )[:add_limit]

    feasible = _build_feasible_moves(current_solution, removable_ranked, candidate_ranked)
    if not feasible:
        return []

    scored: List[Move] = []
    for u, v in feasible:
        base_score = _base_swap_score(u, v, removal_loss, addition_gain)
        tabu = _is_tabu_addition(v, tabu_time, iter_id)
        aspirated = aspiration_baseline is not None and base_score > aspiration_baseline
        if tabu and not aspirated:
            continue

        remove_pressure = core_norm(u)
        add_opportunity = (
            bridge_score.get(v, 0.0) * (1.0 - core_norm(v))
            + frontier_affinity(v)
            + 0.20 * degree_norm(v)
        )
        structural_bonus = core_weight * remove_pressure + add_opportunity
        score = base_score + structural_bonus
        scored.append((score, u, v))

    return _sort_moves_desc(scored)


def neighborhood_elite_path_relinking(
    current_solution: Set[int],
    elite_solution: Set[int] | None,
    removable_vertices: Sequence[int],
    candidate_vertices: Sequence[int],
    removal_loss: Mapping[int, float],
    addition_gain: Mapping[int, float],
    tabu_time: Mapping[int, int],
    iter_id: int,
    rng: random.Random,
    *,
    guidance_weight: float = 0.60,
    perturbation_rate: float = 0.10,
    perturbation_size: int = 5,
    aspiration_baseline: float | None = None,
) -> List[Move]:
    """Elite-guided path relinking neighborhood.

    Logic:
    - Generate guided swaps that move current solution toward an elite solution.
    - Add controlled perturbation moves to avoid over-concentration.

    Trade-off:
    - Strong intensification with a high-quality elite memory.
    - Can over-focus if elite set is poor or too homogeneous.
    """
    _validate_probability(perturbation_rate, "perturbation_rate")
    if perturbation_size < 0:
        raise ValueError("perturbation_size must be non-negative")

    feasible_all = _build_feasible_moves(current_solution, removable_vertices, candidate_vertices)
    if not feasible_all:
        return []

    scored: List[Move] = []

    if elite_solution:
        toward_remove = [u for u in removable_vertices if u in current_solution and u not in elite_solution]
        toward_add = [v for v in candidate_vertices if v not in current_solution and v in elite_solution]
        guided = _build_feasible_moves(current_solution, toward_remove, toward_add)
    else:
        guided = []

    guided_set = set(guided)

    for u, v in guided:
        base_score = _base_swap_score(u, v, removal_loss, addition_gain)
        tabu = _is_tabu_addition(v, tabu_time, iter_id)
        aspirated = aspiration_baseline is not None and base_score > aspiration_baseline
        if tabu and not aspirated:
            continue

        score = base_score + guidance_weight
        scored.append((score, u, v))

    # Controlled random perturbation broadens exploration around relinking path.
    if perturbation_size > 0 and feasible_all:
        non_guided = [pair for pair in feasible_all if pair not in guided_set]
        if non_guided and rng.random() < perturbation_rate:
            draw = min(perturbation_size, len(non_guided))
            sampled_pairs = rng.sample(non_guided, draw)
            for u, v in sampled_pairs:
                base_score = _base_swap_score(u, v, removal_loss, addition_gain)
                tabu = _is_tabu_addition(v, tabu_time, iter_id)
                aspirated = aspiration_baseline is not None and base_score > aspiration_baseline
                if tabu and not aspirated:
                    continue
                scored.append((base_score, u, v))

    if not scored:
        # Fallback to plain ranking if no guided move survives tabu filtering.
        for u, v in feasible_all:
            base_score = _base_swap_score(u, v, removal_loss, addition_gain)
            tabu = _is_tabu_addition(v, tabu_time, iter_id)
            aspirated = aspiration_baseline is not None and base_score > aspiration_baseline
            if tabu and not aspirated:
                continue
            scored.append((base_score, u, v))

    return _sort_moves_desc(scored)


def generate_advanced_neighborhood(
    strategy: str,
    *,
    graph: Graph,
    current_solution: Set[int],
    removable_vertices: Sequence[int],
    candidate_vertices: Sequence[int],
    removal_loss: Mapping[int, float],
    addition_gain: Mapping[int, float],
    tabu_time: Mapping[int, int],
    iter_id: int,
    rng: random.Random,
    stagnation_count: int = 0,
    stagnation_window: int = 20,
    node_frequency: Mapping[int, int] | None = None,
    move_frequency: Mapping[Tuple[int, int], int] | None = None,
    elite_solution: Set[int] | None = None,
    core_number: Mapping[int, int] | None = None,
    bridge_score: Mapping[int, float] | None = None,
    out_degree: Mapping[int, int] | None = None,
    reverse_neighbors: Mapping[int, Set[int]] | None = None,
    aspiration_baseline: float | None = None,
) -> List[Move]:
    """Unified dispatcher for advanced Tabu neighborhoods.

    Supported strategy keys:
    - "rcl_epsilon"
    - "frequency_penalty"
    - "core_periphery"
    - "elite_relinking"
    """
    key = strategy.strip().lower()

    if key == "rcl_epsilon":
        return neighborhood_rcl_epsilon_greedy(
            current_solution=current_solution,
            removable_vertices=removable_vertices,
            candidate_vertices=candidate_vertices,
            removal_loss=removal_loss,
            addition_gain=addition_gain,
            tabu_time=tabu_time,
            iter_id=iter_id,
            rng=rng,
            stagnation_count=stagnation_count,
            stagnation_window=stagnation_window,
            aspiration_baseline=aspiration_baseline,
        )

    if key == "frequency_penalty":
        return neighborhood_frequency_penalized(
            current_solution=current_solution,
            removable_vertices=removable_vertices,
            candidate_vertices=candidate_vertices,
            removal_loss=removal_loss,
            addition_gain=addition_gain,
            tabu_time=tabu_time,
            iter_id=iter_id,
            node_frequency=node_frequency,
            move_frequency=move_frequency,
            aspiration_baseline=aspiration_baseline,
        )

    if key == "core_periphery":
        return neighborhood_structural_core_periphery(
            graph=graph,
            current_solution=current_solution,
            removable_vertices=removable_vertices,
            candidate_vertices=candidate_vertices,
            removal_loss=removal_loss,
            addition_gain=addition_gain,
            tabu_time=tabu_time,
            iter_id=iter_id,
            core_number=core_number,
            bridge_score=bridge_score,
            out_degree=out_degree,
            reverse_neighbors=reverse_neighbors,
            aspiration_baseline=aspiration_baseline,
        )

    if key == "elite_relinking":
        return neighborhood_elite_path_relinking(
            current_solution=current_solution,
            elite_solution=elite_solution,
            removable_vertices=removable_vertices,
            candidate_vertices=candidate_vertices,
            removal_loss=removal_loss,
            addition_gain=addition_gain,
            tabu_time=tabu_time,
            iter_id=iter_id,
            rng=rng,
            aspiration_baseline=aspiration_baseline,
        )

    raise ValueError(
        "Unknown strategy. Use one of: "
        "'rcl_epsilon', 'frequency_penalty', 'core_periphery', 'elite_relinking'."
    )


__all__ = [
    "Graph",
    "ReachabilityGraph",
    "Move",
    "compute_structural_signals",
    "neighborhood_rcl_epsilon_greedy",
    "neighborhood_frequency_penalized",
    "neighborhood_structural_core_periphery",
    "neighborhood_elite_path_relinking",
    "generate_advanced_neighborhood",
]
