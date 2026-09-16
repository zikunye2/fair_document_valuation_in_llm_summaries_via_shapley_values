#!/usr/bin/env python3
"""Main-text figure `fig:benchmark_comparison`: efficiency-accuracy frontiers of Monte
Carlo, Truncated Monte Carlo, Kernel SHAP and Cluster Shapley.

Inputs, all offline:
  outputs/algorithms/baselines/baseline_detailed.csv
      Written by code/algorithms/run_baselines.py --methods mc tmc kernel: one row per
      (method, seed, budget, game) with the number of unique subsets that run evaluated
      and its MAE against the stored document values, as in the manuscript.
  outputs/algorithms/cluster/cluster_summary.csv
      Written by code/algorithms/run_cluster.py: Cluster Shapley with adaptive clustering
      ("separated") on the archived text-embedding-3-large singleton-summary embeddings.

Aggregation, as in the archived research notebook
(Shapley_Value_Baselines_Appendix.ipynb): the runs of each sampling baseline are pooled
over seeds, games and budgets and grouped by the exact number of unique subsets they
evaluated; the curve is the mean MAE of each group. Query membership and weights vary
with the subset count. The shaded bands retain the notebook's pooled t formula as a
descriptive historical display, not a confidence interval across ten independent
seed-level means. Grouping by budget instead (mean unique subsets and mean MAE per budget) gives
a different figure for Truncated Monte Carlo, because games that truncate heavily never
reach large unique-subset counts.

Outputs (outputs/main_text/):
  figure3_benchmark_comparison.pdf and .png
  figure3_benchmark_comparison_points.csv    every plotted point

The points CSV includes the target, query count, aggregation method and historical
band formula for each point. Cluster points average the queries at a fixed epsilon;
baseline points group runs by their actual number of unique subsets.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / "code" / "algorithms"))
from run_baselines import write_csv  # noqa: E402

STYLE = {"cluster": ("Cluster Shapley", "green", "-"),
         "tmc": ("Truncated Monte Carlo", "blue", "--"),
         "mc": ("Monte Carlo", "red", ":"),
         "kernel": ("Kernel SHAP", "purple", "-.")}
LABELLED_EPSILONS = (0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def group_by_unique_subsets(rows):
    """Notebook aggregation: pool all runs and group by the exact unique-subset count."""
    groups = defaultdict(list)
    for r in rows:
        groups[int(round(float(r["unique_subsets"])))].append(float(r["mae"]))
    x = np.array(sorted(groups), dtype=float)
    y = np.array([np.mean(groups[int(k)]) for k in x])
    n = np.array([len(groups[int(k)]) for k in x])
    sd = np.array([np.std(groups[int(k)], ddof=1) if len(groups[int(k)]) > 1 else 0.0 for k in x])
    half = np.where(n > 1, student_t.ppf(0.975, np.maximum(n - 1, 1)) * sd / np.sqrt(n), 0.0)
    return x, y, half, n


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-detailed", type=Path,
                        default=PACKAGE / "outputs/algorithms/baselines/baseline_detailed.csv")
    parser.add_argument("--cluster-summary", type=Path,
                        default=PACKAGE / "outputs/algorithms/cluster/cluster_summary.csv")
    parser.add_argument("--target", default="stored", choices=["stored", "recomputed_exact"])
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "outputs/main_text")
    parser.add_argument("--xlim", type=float, nargs=2, default=(20, 180))
    parser.add_argument("--ylim", type=float, nargs=2, default=(0, 1))
    args = parser.parse_args()

    baseline = [r for r in read_rows(args.baseline_detailed) if r["target"] == args.target]
    cluster = [r for r in read_rows(args.cluster_summary)
               if r["method"] == "separated" and r["target"] == args.target]
    if not baseline or not cluster:
        raise SystemExit("Missing baseline or cluster results; run run_baselines.py and run_cluster.py first")

    fig, ax = plt.subplots(figsize=(10, 6.6))
    points = []
    for method in ["tmc", "mc", "kernel"]:
        rows = [r for r in baseline if r["method"] == method]
        if not rows:
            raise SystemExit(f"Baseline results have no rows for method {method}")
        x, y, half, n = group_by_unique_subsets(rows)
        queries_by_cost = defaultdict(set)
        for r in rows:
            queries_by_cost[int(round(float(r["unique_subsets"])))].add(r["dataset"])
        label, color, style = STYLE[method]
        ax.plot(x, y, color=color, linestyle=style, linewidth=2, label=label)
        ax.fill_between(x, y - half, y + half, color=color, alpha=0.2, linewidth=0)
        for xi, yi, hi, ni in zip(x, y, half, n):
            points.append({"method": label, "target": args.target, "budget": "", "epsilon": "",
                           "unique_subsets": float(xi), "mae": float(yi),
                           "ci95_halfwidth": float(hi), "runs_in_group": int(ni),
                           "query_count": len(queries_by_cost[int(xi)]),
                           "aggregation": "pooled_by_actual_subset_count",
                           "band_method": "historical_pooled_t"})

    cluster = sorted(cluster, key=lambda r: float(r["unique_subsets"]))
    x = np.array([float(r["unique_subsets"]) for r in cluster])
    y = np.array([float(r["mae"]) for r in cluster])
    label, color, style = STYLE["cluster"]
    ax.plot(x, y, color=color, linestyle=style, linewidth=2, label=label)
    for r in cluster:
        epsilon = float(r["initial_epsilon"])
        points.append({"method": label, "target": args.target, "budget": "", "epsilon": epsilon,
                       "unique_subsets": float(r["unique_subsets"]), "mae": float(r["mae"]),
                       "ci95_halfwidth": 0.0, "runs_in_group": 1,
                       "query_count": int(r["query_runs"]),
                       "aggregation": "equal_query_mean_at_epsilon",
                       "band_method": "none"})
        if any(abs(epsilon - e) < 1e-9 for e in LABELLED_EPSILONS):
            ax.plot(float(r["unique_subsets"]), float(r["mae"]), "o", color=color, markersize=6)
            ax.annotate(f"$\\epsilon$={epsilon:g}", (float(r["unique_subsets"]), float(r["mae"])),
                        textcoords="offset points", xytext=(4, -15), fontsize=11, rotation=-15)

    ax.set_xlim(*args.xlim)
    ax.set_ylim(*args.ylim)
    ax.set_xlabel("Number of Unique Subsets", fontsize=14)
    ax.set_ylabel("Mean MAE", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], color=STYLE[m][1], linestyle=STYLE[m][2], linewidth=2, label=STYLE[m][0])
               for m in ["cluster", "tmc", "mc", "kernel"]]
    ax.legend(handles=handles, fontsize=13, frameon=True)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure3_benchmark_comparison.pdf", bbox_inches="tight")
    fig.savefig(out / "figure3_benchmark_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    write_csv(out / "figure3_benchmark_comparison_points.csv", points)
    print(f"Wrote {out / 'figure3_benchmark_comparison.pdf'} with {len(points)} plotted points "
          f"(target: {args.target}; baselines grouped by exact unique-subset count, as in the notebook)")


if __name__ == "__main__":
    main()
