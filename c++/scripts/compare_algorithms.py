#!/usr/bin/env python3
"""Run and summarize degree_heuristic, CELF, and Tabu v3 benchmarks.

The recorded runtime uses "Algorithm core time (exclude P0,P3)" printed by tabu_im,
which excludes subgraph sampling (P0) and final Monte-Carlo evaluation (P3).
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


SPREAD_RE = re.compile(r"Spread:\s*([0-9]+(?:\.[0-9]+)?)")
SEED_SET_RE = re.compile(r"Seed set S:\s*(\[.*\])")
CORE_TIME_RE = re.compile(r"Algorithm core time \(exclude P0,P3\):\s*([0-9]+(?:\.[0-9]+)?)s")


def parse_output(stdout: str) -> Dict[str, object]:
    spread_match = SPREAD_RE.search(stdout)
    seed_set_match = SEED_SET_RE.search(stdout)
    core_time_match = CORE_TIME_RE.search(stdout)

    if seed_set_match is None:
        raise ValueError("Could not parse Seed set S from output")
    if core_time_match is None:
        raise ValueError("Could not parse Algorithm core time from output")

    spread = float(spread_match.group(1)) if spread_match is not None else None

    return {
        "spread": spread,
        "seed_set": seed_set_match.group(1),
        "core_time": float(core_time_match.group(1)),
    }


def run_case(exe_path: Path, cwd: Path, args: List[str]) -> Dict[str, object]:
    process = subprocess.run(
        [str(exe_path), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(
            "Command failed with code "
            f"{process.returncode}: {' '.join(args)}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    parsed = parse_output(process.stdout)
    parsed["raw_stdout"] = process.stdout
    return parsed


def build_algo_args(
    algo: str,
    dataset: str,
    k: int,
    p: float,
    seed: int,
    r_eval: int,
    num_subgraphs: int,
    c: int,
    r: int,
    max_iter: int,
    max_no_improve: int,
    skip_final_eval: bool,
) -> List[str]:
    common = [
        "--algo",
        algo,
        "--dataset",
        dataset,
        "--k",
        str(k),
        "--p",
        str(p),
        "--seed",
        str(seed),
        "--r-eval",
        str(r_eval),
    ]

    if skip_final_eval:
        common.append("--skip-final-eval")

    if algo == "degree_heuristic":
        return common
    if algo == "celf":
        return common + ["--num-subgraphs", str(num_subgraphs)]
    if algo == "tabu_v3":
        return common + [
            "--c",
            str(c),
            "--r",
            str(r),
            "--max-iter",
            str(max_iter),
            "--max-no-improve",
            str(max_no_improve),
        ]

    raise ValueError(f"Unsupported algo: {algo}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare degree_heuristic, CELF, and Tabu v3")
    parser.add_argument("--dataset", default="Wiki-Vote.txt")
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--k-start", type=int, default=10)
    parser.add_argument("--k-end", type=int, default=60)
    parser.add_argument("--k-step", type=int, default=10)
    parser.add_argument("--r-eval", type=int, default=1000)
    parser.add_argument("--num-subgraphs", type=int, default=100)
    parser.add_argument("--c", type=int, default=200)
    parser.add_argument("--r", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--max-no-improve", type=int, default=500)
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Skip final Monte-Carlo spread evaluation (spread will be N/A)",
    )
    parser.add_argument(
        "--output",
        default="c++/results/compare_3_algorithms.md",
        help="Workspace-relative output markdown file",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    exe_path = workspace_root / "c++" / "build-mingw" / "tabu_im.exe"
    run_cwd = exe_path.parent
    output_path = workspace_root / args.output

    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found: {exe_path}")

    ks = list(range(args.k_start, args.k_end + 1, args.k_step))
    algos = ["degree_heuristic", "celf", "tabu_v3"]

    results: Dict[Tuple[str, int], Dict[str, object]] = {}

    for k in ks:
        for algo in algos:
            run_args = build_algo_args(
                algo=algo,
                dataset=args.dataset,
                k=k,
                p=args.p,
                seed=args.seed,
                r_eval=args.r_eval,
                num_subgraphs=args.num_subgraphs,
                c=args.c,
                r=args.r,
                max_iter=args.max_iter,
                max_no_improve=args.max_no_improve,
                skip_final_eval=args.skip_final_eval,
            )
            print(f"Running {algo} with k={k} ...")
            results[(algo, k)] = run_case(exe_path, run_cwd, run_args)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# So sanh 3 thuat toan")
    lines.append("")
    lines.append(f"- Tao luc: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Dataset: {args.dataset}")
    lines.append(f"- p: {args.p}")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- k: {ks}")
    lines.append(f"- num_subgraphs (CELF): {args.num_subgraphs}")
    lines.append(f"- c/r/max_iter/max_no_improve (Tabu): {args.c}/{args.r}/{args.max_iter}/{args.max_no_improve}")
    if args.skip_final_eval:
        lines.append("- final_eval: skipped (--skip-final-eval)")
    else:
        lines.append(f"- final_eval: enabled (R={args.r_eval})")
    lines.append("")
    lines.append("## Bang tong hop")
    lines.append("")
    lines.append("| k | degree spread | degree core time (s) | celf spread | celf core time (s) | tabu_v3 spread | tabu_v3 core time (s) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")

    for k in ks:
        degree = results[("degree_heuristic", k)]
        celf = results[("celf", k)]
        tabu = results[("tabu_v3", k)]
        degree_spread = "N/A" if degree["spread"] is None else f"{degree['spread']:.4f}"
        celf_spread = "N/A" if celf["spread"] is None else f"{celf['spread']:.4f}"
        tabu_spread = "N/A" if tabu["spread"] is None else f"{tabu['spread']:.4f}"
        lines.append(
            "| "
            f"{k} | "
            f"{degree_spread} | {degree['core_time']:.6f} | "
            f"{celf_spread} | {celf['core_time']:.6f} | "
            f"{tabu_spread} | {tabu['core_time']:.6f} |"
        )

    lines.append("")
    lines.append("## Tap nghiem cuoi cung")
    lines.append("")

    for k in ks:
        lines.append(f"### k = {k}")
        for algo in algos:
            item = results[(algo, k)]
            spread_text = "N/A" if item["spread"] is None else f"{item['spread']:.4f}"
            lines.append(f"- {algo}:")
            lines.append(f"  - spread: {spread_text}")
            lines.append(f"  - core_time_s: {item['core_time']:.6f}")
            lines.append(f"  - seed_set: {item['seed_set']}")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report: {output_path}")


if __name__ == "__main__":
    main()
