from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.celf import celf
from src.algorithms.degree_heuristic import degree_heuristic
from src.algorithms.tabu_search import tabu_search
from src.algorithms.tabu_v2 import tabu_search_v2
from src.algorithms.tabu_v3 import tabu_search_v3
from src.common.shared import load_graph, monte_carlo_ic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Influence Maximization Runner")
    parser.add_argument(
        "--algo",
        choices=["celf", "degree", "tabu", "tabu_v2", "tabu_v3"],
        default="celf",
    )
    parser.add_argument("--dataset", default="facebook_combined.txt")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--num-subgraphs", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Shared random seed for algorithm, subgraph sampling, and Monte Carlo evaluation.",
    )
    parser.add_argument(
        "--r-algo",
        type=int,
        default=100,
        help="Deprecated alias for --num-subgraphs when not provided.",
    )
    parser.add_argument(
        "--t-min",
        type=int,
        default=5,
        help="Minimum tabu tenure (inclusive).",
    )
    parser.add_argument(
        "--t-max",
        type=int,
        default=9,
        help="Maximum tabu tenure (inclusive); must satisfy t_max < k.",
    )
    parser.add_argument(
        "--t-tenure",
        type=int,
        default=None,
        help="Deprecated: fixed tabu tenure, overrides --t-min/--t-max.",
    )
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--max-no-improve", type=int, default=1000)
    parser.add_argument("--c", type=int, default=100, help="Tabu candidate pool size.")
    parser.add_argument("--r", type=int, default=100, help="Tabu live-edge subgraphs.")
    parser.add_argument(
        "--tabu-unreached-ratio",
        type=float,
        default=0.80,
        help="Candidate-pool ratio for unreached-focused tier in tabu.",
    )
    parser.add_argument(
        "--tabu-influence-ratio",
        type=float,
        default=0.0,
        help="Candidate-pool ratio for influence tier in tabu.",
    )
    parser.add_argument(
        "--tabu-random-ratio",
        type=float,
        default=0.20,
        help="Candidate-pool ratio for random diversification tier in tabu.",
    )
    parser.add_argument(
        "--neighborhood-strategy",
        type=str,
        default="basic",
        choices=[
            "basic",
            "rcl_epsilon",
            "frequency_penalty",
            "core_periphery",
            "elite_relinking",
        ],
        help="Neighborhood strategy for Tabu Search move ranking.",
    )
    parser.add_argument(
        "--tabu-v3-backtrack-depth",
        type=int,
        default=5,
        help="History depth used by tabu_v3 for deep-backtracking.",
    )
    parser.add_argument(
        "--tabu-v3-backtrack-trigger-ratio",
        type=float,
        default=0.60,
        help="Restart trigger ratio over max-no-improve for tabu_v3.",
    )
    parser.add_argument(
        "--tabu-v3-backtrack-tenure-decay",
        type=float,
        default=0.70,
        help="Tabu-tenure decay applied after deep-backtracking in tabu_v3.",
    )
    parser.add_argument(
        "--tabu-v3-elite-pool-size",
        type=int,
        default=4,
        help="Elite archive size for tabu_v3.",
    )
    parser.add_argument(
        "--tabu-v3-jump-ratio",
        type=float,
        default=0.10,
        help="Fraction of k used for long-jump perturbation in tabu_v3.",
    )
    parser.add_argument(
        "--tabu-v3-restart-policy",
        type=str,
        default="adaptive",
        choices=["stagnation", "periodic", "adaptive"],
        help="Restart policy for tabu_v3.",
    )
    parser.add_argument(
        "--tabu-v3-restart-interval",
        type=int,
        default=50,
        help="Restart interval used by periodic/adaptive tabu_v3 policy.",
    )
    parser.add_argument(
        "--tabu-v3-diversify-random-ratio",
        type=float,
        default=0.30,
        help="Temporary random ratio while tabu_v3 is in diversification mode.",
    )
    parser.add_argument(
        "--tabu-v3-diversify-steps",
        type=int,
        default=3,
        help="Number of iterations to keep diversification mix after a restart.",
    )
    parser.add_argument(
        "--tabu-v3-disable-deep-backtracking",
        action="store_true",
        help="Disable deep-backtracking in tabu_v3 for ablation.",
    )
    parser.add_argument("--r-eval", type=int, default=1000)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    dataset_path = PROJECT_ROOT / "input_txt" / args.dataset

    print("=" * 56)
    print("Influence Maximization - Unified Main")
    print("=" * 56)
    print(f"Algorithm: {args.algo}")
    print(f"Dataset:   {dataset_path.name}")

    graph, vertices = load_graph(dataset_path)
    subgraph_seed = args.seed
    eval_seed = args.seed

    algo_timing = {}
    if args.algo == "celf":
        num_subgraphs = args.num_subgraphs if args.num_subgraphs is not None else args.r_algo

        print("\nRunning CELF...")
        print(f"Live-edge subgraphs: {num_subgraphs}")
        print(f"Seed: {args.seed}")

        seed_set, algo_timing = celf(
            graph,
            vertices,
            k=args.k,
            p=args.p,
            num_subgraphs=num_subgraphs,
            seed=args.seed,
            subgraph_seed=subgraph_seed,
            verbose=True,
        )
    elif args.algo == "tabu":
        t_min = args.t_min
        t_max = args.t_max
        if args.t_tenure is not None:
            t_min = args.t_tenure
            t_max = args.t_tenure

        algo_label = "Tabu Search"
        print(f"\nRunning {algo_label}...")
        print(f"Candidate pool size C: {args.c}")
        print(f"Live-edge subgraphs R: {args.r}")
        print(
            "Candidate mix (unreached/influence/random): "
            f"{args.tabu_unreached_ratio:.2f}/"
            f"{args.tabu_influence_ratio:.2f}/"
            f"{args.tabu_random_ratio:.2f}"
        )
        print("Tabu tenure: T = random(0.4, 0.8) * k")
        print(f"Max iterations: {args.max_iter}")
        print(f"Max no-improve: {args.max_no_improve}")
        print(f"Seed: {args.seed}")

        seed_set, algo_timing = tabu_search(
            graph,
            vertices,
            k=args.k,
            p=args.p,
            c=args.c,
            r=args.r,
            t_min=t_min,
            t_max=t_max,
            max_iter=args.max_iter,
            max_no_improve=args.max_no_improve,
            seed=args.seed,
            subgraph_seed=subgraph_seed,
            global_eval_runs=args.r_eval,
            global_eval_seed=eval_seed,
            unreached_ratio=args.tabu_unreached_ratio,
            influence_ratio=args.tabu_influence_ratio,
            random_ratio=args.tabu_random_ratio,
            verbose=True,
        )
    elif args.algo == "tabu_v2":
        t_min = args.t_min
        t_max = args.t_max
        if args.t_tenure is not None:
            t_min = args.t_tenure
            t_max = args.t_tenure

        algo_label = "Tabu Search v2"
        print(f"\nRunning {algo_label}...")
        print(f"Candidate pool size C: {args.c}")
        print(f"Live-edge subgraphs R: {args.r}")
        print("Tabu tenure: T = random(0.4, 0.8) * k")
        print(f"Neighborhood strategy: {args.neighborhood_strategy}")
        print(f"Max iterations: {args.max_iter}")
        print(f"Max no-improve: {args.max_no_improve}")
        print(f"Seed: {args.seed}")

        seed_set, algo_timing = tabu_search_v2(
            graph,
            vertices,
            k=args.k,
            p=args.p,
            c=args.c,
            r=args.r,
            t_min=t_min,
            t_max=t_max,
            max_iter=args.max_iter,
            max_no_improve=args.max_no_improve,
            neighborhood_strategy=args.neighborhood_strategy,
            seed=args.seed,
            subgraph_seed=subgraph_seed,
            global_eval_runs=args.r_eval,
            global_eval_seed=eval_seed,
            verbose=True,
        )
    elif args.algo == "tabu_v3":
        t_min = args.t_min
        t_max = args.t_max
        if args.t_tenure is not None:
            t_min = args.t_tenure
            t_max = args.t_tenure

        algo_label = "Tabu Search v3"
        print(f"\nRunning {algo_label}...")
        print(f"Candidate pool size C: {args.c}")
        print(f"Live-edge subgraphs R: {args.r}")
        print(
            "Candidate mix (unreached/influence/random): "
            f"{args.tabu_unreached_ratio:.2f}/"
            f"{args.tabu_influence_ratio:.2f}/"
            f"{args.tabu_random_ratio:.2f}"
        )
        print(
            "Deep-backtracking: "
            f"enabled={not args.tabu_v3_disable_deep_backtracking}, "
            f"depth={args.tabu_v3_backtrack_depth}, "
            f"trigger={args.tabu_v3_backtrack_trigger_ratio:.2f}, "
            f"jump={args.tabu_v3_jump_ratio:.2f}"
        )
        print(
            "Restart policy: "
            f"{args.tabu_v3_restart_policy} "
            f"(interval={args.tabu_v3_restart_interval})"
        )
        print(f"Max iterations: {args.max_iter}")
        print(f"Max no-improve: {args.max_no_improve}")
        print(f"Seed: {args.seed}")

        seed_set, algo_timing = tabu_search_v3(
            graph,
            vertices,
            k=args.k,
            p=args.p,
            c=args.c,
            r=args.r,
            t_min=t_min,
            t_max=t_max,
            max_iter=args.max_iter,
            max_no_improve=args.max_no_improve,
            seed=args.seed,
            subgraph_seed=subgraph_seed,
            global_eval_runs=args.r_eval,
            global_eval_seed=eval_seed,
            unreached_ratio=args.tabu_unreached_ratio,
            influence_ratio=args.tabu_influence_ratio,
            random_ratio=args.tabu_random_ratio,
            use_deep_backtracking=not args.tabu_v3_disable_deep_backtracking,
            backtrack_depth=args.tabu_v3_backtrack_depth,
            backtrack_trigger_ratio=args.tabu_v3_backtrack_trigger_ratio,
            backtrack_tenure_decay=args.tabu_v3_backtrack_tenure_decay,
            elite_pool_size=args.tabu_v3_elite_pool_size,
            jump_ratio=args.tabu_v3_jump_ratio,
            restart_policy=args.tabu_v3_restart_policy,
            restart_interval=args.tabu_v3_restart_interval,
            diversify_random_ratio=args.tabu_v3_diversify_random_ratio,
            diversify_steps=args.tabu_v3_diversify_steps,
            verbose=True,
        )
    else:
        print("\nRunning Degree Heuristic...")
        seed_set = set(degree_heuristic(graph, args.k))

    print(f"\nEvaluating spread with Monte Carlo (R={args.r_eval})...")
    if eval_seed is not None:
        print(f"Evaluation seed: {eval_seed}")
    t_eval_start = time.perf_counter()
    spread = monte_carlo_ic(graph, seed_set, p=args.p, r=args.r_eval, seed=eval_seed)
    eval_time = time.perf_counter() - t_eval_start

    print("\n" + "=" * 56)
    print("Result")
    print(f"Seed set S: {sorted(seed_set)}")
    print(f"k:          {args.k}")
    print(f"p:          {args.p}")
    print(f"Spread:     {spread:.4f}")

    print("\nTiming")
    if args.algo == "celf":
        print(f"P0 sample subgraphs:{algo_timing.get('phase_sample_subgraphs', 0.0):10.4f}s")
        print(f"P1 queue init:      {algo_timing.get('phase_init_queue', 0.0):.4f}s")
        print(f"P2 k-loop total:    {algo_timing.get('phase_k_loop', 0.0):.4f}s")
        print(f"   recompute time:  {algo_timing.get('phase_recompute_only', 0.0):.4f}s")
        print(f"   recompute count: {algo_timing.get('stale_recompute_count', 0)}")
    elif args.algo in {"tabu", "tabu_v2", "tabu_v3"}:
        print(f"P0 sample subgraphs:{algo_timing.get('phase_sample_subgraphs', 0.0):10.4f}s")
        print(f"P1 initialize:      {algo_timing.get('phase_init', 0.0):.4f}s")
        print(f"P2 tabu search:     {algo_timing.get('phase_search', 0.0):.4f}s")
        print(f"   iterations:      {algo_timing.get('iterations_completed', 0)}")
        print(f"   neighbor evals:  {algo_timing.get('neighbor_evaluations', 0)}")
        print(f"   global best:     {algo_timing.get('global_best_spread', 0.0):.4f}")
        print(f"   no-improve cnt:  {algo_timing.get('final_no_improve_count', 0)}")
        print(f"   stop no move:    {algo_timing.get('stopped_no_valid_move', 0)}")
        if args.algo == "tabu_v3":
            print(f"   deep backtracks: {algo_timing.get('deep_backtracks', 0)}")
            print(f"   cycle hits:      {algo_timing.get('cycle_hits', 0)}")
            print(f"   periodic restart:{algo_timing.get('periodic_restarts', 0)}")
            print(f"   diversify iters: {algo_timing.get('diversify_iters', 0)}")
            print(f"   elite size max:  {algo_timing.get('elite_archive_size_max', 0)}")
    print(f"P3 final eval:      {eval_time:.4f}s")
    print("=" * 56)


if __name__ == "__main__":
    main()
