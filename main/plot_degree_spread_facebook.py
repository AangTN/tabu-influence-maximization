from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.degree_heuristic import degree_heuristic
from src.common.shared import load_graph, monte_carlo_ic

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("Matplotlib is required. Install with: pip install matplotlib") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Degree-based Heuristic on facebook_combined for k in [1, 30], "
            "estimate spread, and plot a line chart."
        )
    )
    parser.add_argument("--dataset", default="facebook_combined.txt")
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=30)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--r", type=int, default=1000, help="Monte Carlo runs")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for Monte Carlo")
    parser.add_argument(
        "--output-csv",
        default="results/facebook_degree_spread_k1_30.csv",
        help="Workspace-relative path for (k, spread) output",
    )
    parser.add_argument(
        "--output-plot",
        default="results/facebook_degree_spread_k1_30.png",
        help="Workspace-relative path for line chart",
    )
    parser.add_argument("--show", action="store_true", help="Show chart window after saving")
    return parser


def resolve_dataset_path(dataset_arg: str) -> Path:
    direct = Path(dataset_arg)
    if direct.exists():
        return direct

    candidate = PROJECT_ROOT / "input_txt" / dataset_arg
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Dataset not found: {dataset_arg}")


def main() -> None:
    args = build_parser().parse_args()
    if args.k_min <= 0:
        raise ValueError("k-min must be >= 1")
    if args.k_max < args.k_min:
        raise ValueError("k-max must be >= k-min")

    dataset_path = resolve_dataset_path(args.dataset)
    graph, _ = load_graph(dataset_path, directed=False)

    ks: list[int] = []
    spreads: list[float] = []

    for k in range(args.k_min, args.k_max + 1):
        seed_set = degree_heuristic(graph, k)
        spread = monte_carlo_ic(graph, seed_set, p=args.p, r=args.r, seed=args.seed)
        ks.append(k)
        spreads.append(spread)
        print(f"k={k:2d} -> spread={spread:.4f}")

    csv_path = PROJECT_ROOT / args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "spread"])
        for k, spread in zip(ks, spreads):
            writer.writerow([k, f"{spread:.6f}"])

    plot_path = PROJECT_ROOT / args.output_plot
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(ks, spreads, marker="o", linewidth=2)
    plt.xlabel(r"Kích thước tập hạt giống $|S|$")
    plt.ylabel("Độ lan truyền ảnh hưởng")
    plt.title(f"Degree-based Heuristic trên facebook_combined (p={args.p})")
    plt.grid(True, linestyle="--", alpha=0.35)
    x_tick_start = args.k_min if args.k_min % 2 == 0 else args.k_min + 1
    x_ticks = list(range(x_tick_start, args.k_max + 1, 2))
    if x_ticks:
        plt.xticks(x_ticks)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
