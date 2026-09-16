#!/usr/bin/env python3
"""Cluster Shapley from explicitly supplied, provenance-labelled distances.

The default replication uses the text-embedding-3-large singleton-summary embeddings
collected on 2026-09-06 (data/embeddings/main_8doc_summaries, converted to cosine
distances by prepare_distances.py). The original embedding cache was not archived; the
new collection reproduces the archived notebook run's clusterings for 99.4% of the
(game, epsilon) pairs. This executable never silently uses the small-model assumption
cache. No model API calls are made. See README.md for the required input manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from run_baselines import PACKAGE, exact_shapley, load_game, metrics, monte_carlo, write_csv

# The manuscript's 41-value grid: 0.01 and 0.025, 0.05, ..., 1.0.
EPSILONS = [0.01] + [round(i / 40, 3) for i in range(1, 41)]


def components(distances, epsilon):
    """DBSCAN(min_samples=1, precomputed distances) equals graph components."""
    n = len(distances)
    labels = np.full(n, -1, dtype=int)
    label = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        labels[start] = label
        pending = [start]
        while pending:
            i = pending.pop()
            for j in np.flatnonzero(distances[i] <= epsilon):
                if labels[j] == -1:
                    labels[j] = label
                    pending.append(int(j))
        label += 1
    return labels


def separated_clusters(distances, initial_epsilon, max_iterations=100):
    """Notebook separation rule: shrink epsilon by .95 until all pairs comply."""
    epsilon = initial_epsilon
    for iteration in range(max_iterations):
        labels = components(distances, epsilon)
        same = labels[:, None] == labels[None, :]
        if np.array_equal(same, distances <= epsilon):
            return labels, epsilon, iteration + 1
        epsilon *= 0.95
    raise ValueError("No separated partition after 100 iterations; the legacy notebook returned its last invalid partition here")


def quotient_game(values, labels):
    clusters = [np.flatnonzero(labels == label).tolist() for label in sorted(set(labels))]
    masks = [sum(1 << i for i in indices) for indices in clusters]
    reduced = np.zeros(1 << len(clusters))
    for coalition in range(1, len(reduced)):
        document_mask = 0
        for j, mask in enumerate(masks):
            if coalition & (1 << j):
                document_mask |= mask
        reduced[coalition] = values[document_mask]
    return reduced, clusters


def distribute(cluster_values, clusters, n):
    result = np.zeros(n)
    for value, indices in zip(cluster_values, clusters):
        result[indices] = value / len(indices)
    return result


def load_distances(manifest_path, allow_alternative_model=False):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("input_text") != "singleton summaries":
        raise ValueError("Manifest input_text must identify singleton summaries")
    is_main = manifest.get("model") == "text-embedding-3-large" and manifest.get("dimensions") == 3072
    if not is_main and not allow_alternative_model:
        raise ValueError("Main experiment requires text-embedding-3-large / 3072. Alternative models require --allow-alternative-model and are sensitivity runs")
    arrays = {}
    for dataset, relative in manifest["files"].items():
        path = manifest_path.parent / relative
        if path.suffix != ".npy":
            raise ValueError("Distance files must be numeric .npy arrays")
        matrix = np.load(path, allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{dataset}: invalid matrix dimensions or values")
        if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-10) or not np.allclose(np.diag(matrix), 0, atol=1e-10):
            raise ValueError(f"{dataset}: matrix must be symmetric with zero diagonal")
        if np.min(matrix) < -1e-10 or np.max(matrix) > 2+1e-10:
            raise ValueError(f"{dataset}: cosine distance outside [0,2]")
        arrays[dataset] = np.maximum(matrix, 0)
    return manifest, arrays, is_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-manifest", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE / "data/coalition_games_8doc")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "outputs/algorithms/cluster")
    parser.add_argument("--allow-alternative-model", action="store_true")
    parser.add_argument("--epsilons", nargs="+", type=float, default=EPSILONS)
    parser.add_argument("--cluster-permutations", nargs="*", type=int, default=[],
                        help="Optional cluster-level Monte Carlo budgets, e.g. 10 20 100")
    parser.add_argument("--replications", type=int, default=10)
    args = parser.parse_args()
    if any(e <= 0 for e in args.epsilons) or any(k <= 0 for k in args.cluster_permutations) or args.replications < 1:
        parser.error("Epsilons, budgets and replications must be positive")
    manifest, arrays, is_main = load_distances(args.distance_manifest.resolve(), args.allow_alternative_model)
    paths = sorted(args.data_dir.resolve().glob("*.csv"))
    if not paths or set(arrays) != {path.name for path in paths}:
        parser.error("Manifest must map every input CSV filename exactly once")
    rows, assignments = [], []
    for path in paths:
        values, stored = load_game(path)
        exact = exact_shapley(values)
        n = len(stored)
        distances = arrays[path.name]
        if distances.shape != (n, n):
            raise ValueError(f"{path.name}: distance size does not match document count")
        for epsilon in args.epsilons:
            for method in ["separated", "standard_dbscan"]:
                if method == "separated":
                    labels, actual_epsilon, iterations = separated_clusters(distances, epsilon)
                else:
                    labels, actual_epsilon, iterations = components(distances, epsilon), epsilon, 1
                reduced, clusters = quotient_game(values, labels)
                phi = distribute(exact_shapley(reduced), clusters, n)
                assignments.append({"dataset": path.name, "method": method, "initial_epsilon": epsilon,
                                    "actual_epsilon": actual_epsilon, "iterations": iterations,
                                    "clusters": json.dumps(clusters), "num_clusters": len(clusters)})
                for target_name, target in [("stored", stored), ("recomputed_exact", exact)]:
                    rows.append({"dataset": path.name, "method": method, "initial_epsilon": epsilon,
                                 "actual_epsilon": actual_epsilon, "seed": -1, "budget": 0,
                                 "num_clusters": len(clusters), "unique_subsets": len(reduced)-1,
                                 "normalized_cost": (len(reduced)-1)/(len(values)-1),
                                 "target": target_name, **metrics(target, phi)})
                if method == "separated":
                    for count in args.cluster_permutations:
                        for seed in range(args.replications):
                            cluster_phi, used, _ = monte_carlo(reduced, count, random.Random(seed))
                            estimate = distribute(cluster_phi, clusters, n)
                            for target_name, target in [("stored", stored), ("recomputed_exact", exact)]:
                                rows.append({"dataset": path.name, "method": "separated_cluster_mc",
                                             "initial_epsilon": epsilon, "actual_epsilon": actual_epsilon,
                                             "seed": seed, "budget": count, "num_clusters": len(clusters),
                                             "unique_subsets": used, "normalized_cost": used/(len(values)-1),
                                             "target": target_name, **metrics(target, estimate)})
    out = args.output_dir.resolve()
    write_csv(out / "cluster_detailed.csv", rows)
    write_csv(out / "cluster_assignments.csv", assignments)
    groups = {}
    for row in rows:
        key = (row["method"], row["initial_epsilon"], row["budget"], row["target"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (method, epsilon, budget, target), group in groups.items():
        summary.append({"method": method, "initial_epsilon": epsilon, "budget": budget, "target": target,
                        "query_runs": len(group), **{metric: float(np.mean([r[metric] for r in group]))
                         for metric in ["mae", "mse", "mape", "num_clusters", "unique_subsets", "normalized_cost"]}})
    write_csv(out / "cluster_summary.csv", summary)
    metadata = {"distance_model": manifest["model"], "dimensions": manifest["dimensions"],
                "uses_main_model_specification": is_main,
                "historical_cache_provenance_verified": False,
                "interpretation": "Offline algorithm rerun from user-supplied distances; a model match alone does not establish historical identity.",
                "cluster_mc_rng": "Separate random.Random(seed) for each query, epsilon and budget; original global order was not archived."}
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Wrote {len(rows)} cluster comparisons to {out}")


if __name__ == "__main__":
    main()
