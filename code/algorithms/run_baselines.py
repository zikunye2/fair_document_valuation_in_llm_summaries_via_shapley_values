#!/usr/bin/env python3
"""Offline baseline analyses of the archived coalition-value games.

Preserves the MC/TMC/Kernel SHAP configurations of the original baseline
notebook. Stored targets and mathematical exact values are reported separately.
No model calls, credentials, notebook execution, or downloaded data are used.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
import warnings

import numpy as np

PACKAGE = Path(__file__).resolve().parents[2]
MC_COUNTS = list(range(1, 11)) + list(range(12, 81, 2)) + [90, 100, 125, 150, 175, 200, 250, 300, 500, 800, 1000]
TMC_COUNTS = [1, 2, 3, 5, 8, 9, 10] + list(range(12, 81, 2)) + [90, 100, 125, 150, 175, 200, 250, 300, 500, 800, 1000]
KERNEL_COUNTS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 130, 150, 170, 190, 210, 230, 250, 300, 350, 400, 500, 600, 700, 900, 1100]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_game(path):
    """Parse subset labels as strings, retaining leading zero document IDs."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    stored = {}
    subsets = {}
    for row in rows:
        label = row["subset"].strip()
        value = float(row["avg_score"])
        if not np.isfinite(value):
            raise ValueError(f"{path.name}: nonfinite score")
        if label.startswith("Comment "):
            i = int(label.removeprefix("Comment "))
            if i in stored:
                raise ValueError(f"{path.name}: duplicate attribution row")
            stored[i] = value
        elif label.isdigit():
            ids = [int(i) for i in label]
            if ids != sorted(set(ids)):
                raise ValueError(f"{path.name}: malformed subset label {label}")
            mask = sum(1 << i for i in ids)
            if mask in subsets:
                raise ValueError(f"{path.name}: duplicate coalition")
            subsets[mask] = value
        else:
            raise ValueError(f"{path.name}: unknown subset label")
    n = len(stored)
    if not 1 <= n <= 10 or set(stored) != set(range(n)):
        raise ValueError(f"{path.name}: document IDs must be consecutive, n <= 10")
    if set(subsets) != set(range(1, 1 << n)):
        raise ValueError(f"{path.name}: incomplete coalition game")
    values = np.zeros(1 << n, dtype=float)
    for mask, score in subsets.items():
        values[mask] = score
    return values, np.array([stored[i] for i in range(n)])


def exact_shapley(values):
    n = (len(values) - 1).bit_length()
    phi = np.zeros(n)
    for i in range(n):
        bit = 1 << i
        phi[i] = math.fsum(
            (values[mask | bit] - values[mask]) / (n * math.comb(n - 1, mask.bit_count()))
            for mask in range(len(values)) if not mask & bit
        )
    return phi


def metrics(target, estimate):
    error = target - estimate
    denominator = target + 0.1  # Original notebook's stabilized MAPE.
    if np.any(denominator == 0):
        raise ValueError("MAPE denominator is zero; no value was silently replaced")
    mse = float(np.mean(error ** 2))
    return {"mae": float(np.mean(np.abs(error))), "mse": mse,
            "rmse": math.sqrt(mse), "mape": float(np.mean(np.abs(error / denominator)) * 100)}


def monte_carlo(values, count, py_rng):
    """Uniform random.shuffle permutations with replacement, as in cell 6."""
    n = (len(values) - 1).bit_length()
    phi = np.zeros(n)
    used = set()
    for _ in range(count):
        permutation = list(range(n))
        py_rng.shuffle(permutation)
        previous, mask = 0.0, 0
        for i in permutation:
            mask |= 1 << i
            used.add(mask)
            phi[i] += values[mask] - previous
            previous = values[mask]
    phi /= count
    # The notebook normalizes to the full-set score (normally a no-op by
    # telescoping). Preserve its zero-sum convention explicitly.
    total = phi.sum()
    phi = phi / (total if total != 0 else 1e-6) * values[-1]
    return phi, len(used), count * n


def truncated_monte_carlo(values, count, np_rng, tolerance=0.5):
    """Legacy notebook TMC: uses the best score seen, not a fixed ceiling 10.

    best_score_seen persists across permutations within this run. The original
    notebook's criterion is retained because changing it changes the experiment.
    """
    n = (len(values) - 1).bit_length()
    phi, best = np.zeros(n), None
    used, evaluations = set(), 0
    for t in range(count):
        current, mask = 0.0, 0
        marginal = np.zeros(n)
        for i in np_rng.permutation(n):
            old = current
            mask |= 1 << int(i)
            if best is None or best - old >= tolerance:
                current = values[mask]
                used.add(mask)
                evaluations += 1
                if best is None or current > best:
                    best = current
            marginal[i] = current - old
        phi = (t * phi + marginal) / (t + 1)
    return phi, len(used), evaluations


def kernel_shap(values, count, warning_sink=None):
    """Original SHAP KernelExplainer settings; requires the optional shap package."""
    import shap
    n = (len(values) - 1).bit_length()
    used, calls = set(), [0]

    def lookup(x):
        masks = np.sum((np.asarray(x) == 1).astype(np.int64) * (1 << np.arange(n)), axis=1)
        used.update(int(mask) for mask in masks)
        calls[0] += len(masks)
        return values[masks]

    explainer = shap.KernelExplainer(lookup, np.zeros((1, n)), link="identity",
                                   l1_reg="aic", feature_perturbation="interventional", model_output="raw")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = explainer.shap_values(np.ones((1, n)), nsamples=count, l1_reg="aic", silent=True)
    if warning_sink is not None:
        warning_sink.extend({"category": item.category.__name__, "message": str(item.message)} for item in caught)
    elif caught:
        warnings.warn(f"Kernel SHAP emitted {len(caught)} numerical warnings; supply warning_sink to record them", RuntimeWarning)
    if isinstance(result, list):
        result = result[0]
    return np.asarray(result).reshape(-1), len(used), calls[0]


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["budget"], row["target"]), []).append(row)
    summary = []
    for (method, budget, target), group in groups.items():
        seeds = sorted({row["seed"] for row in group})
        record = {"method": method, "budget": budget, "target": target,
                  "replications": len(seeds), "query_runs": len(group)}
        for metric in ["mae", "mse", "rmse", "mape", "unique_subsets", "total_subsets"]:
            run_means = [np.mean([row[metric] for row in group if row["seed"] == seed]) for seed in seeds]
            record[metric] = float(np.mean(run_means))
            record[metric + "_sd_across_runs"] = float(np.std(run_means, ddof=1)) if len(seeds) > 1 else 0.0
        summary.append(record)
    return summary


def plot_frontiers(summary, path):
    """Optional publication-style comparison plot; never inserts missing curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import t
    rows = [r for r in summary if r["target"] == "stored"]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, metric in zip(axes, ["mae", "mse", "mape"]):
        for method in methods:
            points = sorted([r for r in rows if r["method"] == method], key=lambda r: r["unique_subsets"])
            x = np.array([r["unique_subsets"] for r in points])
            y = np.array([r[metric] for r in points])
            axis.plot(x, y, label=method)
            ci = np.array([t.ppf(.975, r["replications"] - 1) * r[metric + "_sd_across_runs"] / math.sqrt(r["replications"])
                           if r["replications"] > 1 else 0.0 for r in points])
            axis.fill_between(x, y-ci, y+ci, alpha=.14)
        axis.set(xlabel="Unique subsets evaluated", ylabel=metric.upper())
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Offline baseline rerun; historical stored targets")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE / "data/coalition_games_8doc")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "outputs/algorithms/baselines")
    parser.add_argument("--quick", action="store_true", help="2 queries, 2 seeds, 3 budgets; a smoke run, not paper results")
    parser.add_argument("--audit-only", action="store_true", help="Audit every supplied game and compute exact/equal attribution only")
    parser.add_argument("--methods", nargs="+", choices=["mc", "tmc", "kernel"], default=["mc", "tmc"])
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--plot", action="store_true", help="Produce PDF curves; requires matplotlib and scipy")
    args = parser.parse_args()
    # SHAP imports plotting modules. Give optional plotting dependencies a
    # temporary writable cache without writing into a reviewer's home directory.
    plot_cache = tempfile.TemporaryDirectory(prefix="shapley-plot-cache-")
    os.environ.setdefault("MPLCONFIGDIR", plot_cache.name)
    os.environ.setdefault("XDG_CACHE_HOME", plot_cache.name)
    if args.replications < 1:
        parser.error("--replications must be positive")
    paths = sorted(args.data_dir.resolve().glob("*.csv"))
    if not paths:
        parser.error("No input CSV files found")
    if args.quick:
        paths = paths[:2]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    games, audits, attribution, equal = [], [], [], []
    for path in paths:
        values, stored = load_game(path)
        exact = exact_shapley(values)
        n = len(stored)
        if not np.isclose(exact.sum(), values[-1], rtol=0, atol=1e-10):
            raise ValueError(f"{path.name}: exact efficiency invariant failed")
        games.append((path.name, values, stored, exact))
        audits.append({"dataset": path.name, "documents": n, "coalitions": len(values)-1,
                       "full_value": float(values[-1]), "stored_sum": float(stored.sum()),
                       "exact_sum": float(exact.sum()), "max_stored_exact_difference": float(np.max(np.abs(stored-exact))),
                       "matches_exact_at_1e-8": bool(np.allclose(stored, exact, rtol=0, atol=1e-8))})
        for i in range(n):
            attribution.append({"dataset": path.name, "document": i, "stored": float(stored[i]),
                                "recomputed_exact": float(exact[i]), "difference": float(stored[i]-exact[i])})
        for target_name, target in [("stored", stored), ("recomputed_exact", exact)]:
            equal_error = target - np.full(n, values[-1]/n)
            equal_mse = float(np.mean(equal_error ** 2))
            equal.append({"dataset": path.name, "target": target_name,
                          "mae": float(np.mean(np.abs(equal_error))),
                          "mse": equal_mse, "rmse": math.sqrt(equal_mse)})
    write_csv(out / "coalition_audit.csv", audits)
    write_csv(out / "exact_attributions.csv", attribution)
    write_csv(out / "equal_attribution_by_query.csv", equal)
    write_csv(out / "equal_attribution_summary.csv", [
        {"target": target, "queries": len(games), **{metric: float(np.mean([row[metric] for row in equal if row["target"] == target]))
         for metric in ["mae", "mse", "rmse"]}} for target in ["stored", "recomputed_exact"]])
    mismatches = sum(not row["matches_exact_at_1e-8"] for row in audits)
    print(f"Audited {len(games)} games; stored/exact mismatches: {mismatches}", flush=True)
    started = time.monotonic()
    rows, numerical_warnings = [], []
    if not args.audit_only:
        counts = {"mc": MC_COUNTS, "tmc": TMC_COUNTS, "kernel": KERNEL_COUNTS}
        for method in args.methods:
            if method == "kernel":
                import shap  # fail visibly before a long loop if missing
            for seed in range(2 if args.quick else args.replications):
                py_rng = random.Random(seed)
                np_rng = np.random.RandomState(seed)
                np.random.seed(seed)  # SHAP consumes numpy's legacy global RNG.
                for count in ([5, 20, 100] if args.quick else counts[method]):
                    for name, values, stored, exact in games:
                        if method == "mc":
                            phi, unique, total = monte_carlo(values, count, py_rng)
                        elif method == "tmc":
                            phi, unique, total = truncated_monte_carlo(values, count, np_rng)
                        else:
                            caught = []
                            phi, unique, total = kernel_shap(values, count, caught)
                            for warning in caught:
                                numerical_warnings.append({"seed": seed, "budget": count, "dataset": name, **warning})
                        if not np.all(np.isfinite(phi)):
                            raise ValueError(f"Nonfinite {method} estimates")
                        for target_name, target in [("stored", stored), ("recomputed_exact", exact)]:
                            rows.append({"method": method, "seed": seed, "budget": count, "dataset": name,
                                         "target": target_name, "unique_subsets": unique, "total_subsets": total,
                                         **metrics(target, phi)})
                print(f"Completed {method}, seed {seed}", flush=True)
        write_csv(out / "baseline_detailed.csv", rows)
        if numerical_warnings:
            write_csv(out / "kernel_numerical_warnings.csv", numerical_warnings)
            print(f"Recorded {len(numerical_warnings)} Kernel SHAP numerical warnings in kernel_numerical_warnings.csv", flush=True)
        summary = summarize(rows)
        write_csv(out / "baseline_summary.csv", summary)
        if args.plot:
            plot_frontiers(summary, out / "baseline_comparison.pdf")
    versions = {}
    for package in ["numpy", "shap", "scikit-learn", "scipy", "matplotlib"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {"quick_smoke_run": args.quick, "audit_only": args.audit_only,
                "dataset_order": [p.name for p in paths], "methods": [] if args.audit_only else args.methods,
                "replications": 2 if args.quick else args.replications, "versions": versions,
                "runtime_seconds": time.monotonic()-started, "stored_exact_mismatches": mismatches,
                "kernel_numerical_warning_count": len(numerical_warnings),
                "disclosures": ["Stored targets are retained; recomputed mathematical exact targets are separate.",
                                "TMC uses best_score_seen stopping with tolerance 0.5.",
                                "Files are processed in sorted filename order.",
                                "Kernel SHAP unique-subset count includes the empty set."]}
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(f"Outputs: {out}", flush=True)


if __name__ == "__main__":
    main()
