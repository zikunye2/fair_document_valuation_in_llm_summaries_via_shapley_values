#!/usr/bin/env python3
"""Main-text figure `fig:clustering_vis`: the eight reviews of the wireless-controller example,
their clusters, and their exact and Cluster Shapley values.

Procedure of the published figure, reproduced here offline:
  1. Each review is represented by the two-dimensional embedding of "review title. single-review
     summary" obtained from the OpenAI API (text-embedding-3-large, dimensions=2). The eight
     vectors are stored in data/embeddings/figure2_example_2d/embeddings_2d.json
     (collected by code/generation/embed_figure2_example.py).
  2. DBSCAN (min_samples=1, eps=0.05) on the cosine distances of the two-dimensional vectors
     gives the clusters (standard DBSCAN with min_samples=1 is the connected-component
     clustering of the distance graph).
  3. The exact Shapley values come from the coalition scores of the game; the Cluster Shapley
     values from the cluster-level game (union coalitions) with an equal split inside clusters.
  4. For display, the vectors are min-max scaled to [-5, 5] on each axis.

Inputs, all offline:
  --game            coalition game CSV (default: the archived example game in
                    data/coalition_games_8doc/)
  --embeddings      data/embeddings/figure2_example_2d/embeddings_2d.json

Outputs (outputs/main_text/, names set by --output-name):
  <name>.pdf and .png
  <name>_points.csv   scaled coordinates, cluster labels, exact and Cluster Shapley values
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / "code" / "algorithms"))
from run_baselines import exact_shapley, load_game, write_csv  # noqa: E402
from run_cluster import distribute, quotient_game  # noqa: E402

DEFAULT_DATASET = "Shapley_the quality of the product_switch.csv"
DEFAULT_EMBEDDINGS = PACKAGE / "data" / "embeddings" / "figure2_example_2d" / "embeddings_2d.json"
# Cluster colours of the published figure, by cluster index in order of first appearance.
COLORS = ["#2F4B9A", "#A6D98B", "#F4A3A3", "#C98BE6", "#B8860B", "#8FB0E8", "#999999", "#444444"]


def dbscan_clusters(vectors, epsilon, min_samples=1):
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    distances = np.clip(1 - unit @ unit.T, 0, 2)
    labels = DBSCAN(eps=epsilon, min_samples=min_samples, metric="precomputed").fit_predict(distances)
    if np.any(labels < 0):
        raise ValueError("DBSCAN marked a review as noise; with min_samples=1 this cannot happen")
    return labels, distances


def scale_for_display(vectors, span=10.0):
    low, high = vectors.min(axis=0), vectors.max(axis=0)
    return (vectors - low) / (high - low) * span - span / 2


def draw(vectors, labels, exact, approx, epsilon, output, title=None):
    coords = scale_for_display(vectors)
    fig, ax = plt.subplots(figsize=(8, 8))
    boxes = {}
    for label in sorted(set(labels)):
        members = np.flatnonzero(labels == label)
        color = COLORS[label % len(COLORS)]
        pad = 0.5
        lo = coords[members].min(axis=0) - pad
        hi = coords[members].max(axis=0) + pad
        boxes[label] = (lo, hi)
        ax.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, edgecolor=color, linewidth=2))
        ax.scatter(coords[members, 0], coords[members, 1], c=[color], s=150, edgecolors="black", zorder=3)
    # Label positions: beside the cluster box; reviews of one cluster with nearly the same
    # height get stacked labels so that they do not overlap.
    label_y = coords[:, 1].copy()
    for label in sorted(set(labels)):
        members = list(np.flatnonzero(labels == label))
        members.sort(key=lambda i: coords[i, 1], reverse=True)
        for a, b in zip(members, members[1:]):
            if label_y[a] - label_y[b] < 1.1:
                mid = (label_y[a] + label_y[b]) / 2
                label_y[a], label_y[b] = mid + 0.55, mid - 0.55
    def blocked(x0, x1, y0, y1, own):
        return any(o != own and b_lo[0] < x1 and b_hi[0] > x0 and b_lo[1] < y1 and b_hi[1] > y0
                   for o, (b_lo, b_hi) in boxes.items())

    for i in range(len(vectors)):
        lo, hi = boxes[labels[i]]
        right = hi[0] + 2.6 <= 6 and not blocked(hi[0], hi[0] + 2.6, label_y[i] - 0.6, label_y[i] + 0.6, labels[i])
        x = hi[0] + 0.25 if right else lo[0] - 0.25
        ax.text(x, label_y[i], f"$\\phi_{{{i + 1}}}$ = {exact[i]:.2f}\n$\\hat{{\\phi}}_{{{i + 1}}}$ = {approx[i]:.2f}",
                fontsize=12, ha="left" if right else "right", va="center")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return coords


def make_figure(game_path, embeddings_path, epsilon, output_dir, output_name, expected_clusters=None):
    values, _ = load_game(Path(game_path))
    n = (len(values) - 1).bit_length()
    record = json.loads(Path(embeddings_path).read_text(encoding="utf-8"))
    vectors = np.array(record["embeddings"], dtype=float)
    if vectors.shape != (n, 2):
        raise ValueError(f"Expected {n} two-dimensional embeddings, found {vectors.shape}")
    labels, distances = dbscan_clusters(vectors, epsilon)
    clusters = [[int(i) + 1 for i in np.flatnonzero(labels == label)] for label in sorted(set(labels))]
    if expected_clusters is not None and clusters != expected_clusters:
        raise ValueError(f"DBSCAN clusters {clusters} differ from the expected {expected_clusters}")
    exact = exact_shapley(values)
    reduced, cluster_list = quotient_game(values, labels)
    approx = distribute(exact_shapley(reduced), cluster_list, n)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    coords = draw(vectors, labels, exact, approx, epsilon, out / output_name)
    write_csv(out / f"{output_name}_points.csv",
              [{"document_one_indexed": i + 1, "cluster": int(labels[i]) + 1,
                "embedding_1": float(vectors[i, 0]), "embedding_2": float(vectors[i, 1]),
                "display_x": float(coords[i, 0]), "display_y": float(coords[i, 1]),
                "exact_shapley": float(exact[i]), "cluster_shapley": float(approx[i])} for i in range(n)])
    print(f"{Path(game_path).name}: {len(clusters)} clusters at epsilon {epsilon:g}: {clusters}")
    print("exact Shapley  :", [round(float(x), 2) for x in exact])
    print("Cluster Shapley:", [round(float(x), 2) for x in approx])
    print(f"Wrote {out / output_name}.pdf")
    return clusters, exact, approx


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", type=Path, default=PACKAGE / "data/coalition_games_8doc" / DEFAULT_DATASET)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "outputs/main_text")
    parser.add_argument("--output-name", default="figure2_clustering_visualization")
    args = parser.parse_args()
    make_figure(args.game, args.embeddings, args.epsilon, args.output_dir, args.output_name)


if __name__ == "__main__":
    main()
