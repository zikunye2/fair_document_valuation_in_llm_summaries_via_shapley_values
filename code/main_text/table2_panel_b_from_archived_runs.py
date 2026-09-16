#!/usr/bin/env python3
"""Table 2 Panel B from the archived Cluster Shapley summary (data/archived_runs/).

data/archived_runs/table2_panel_b/metrics_analysis.csv is the summary of the original
adaptive Cluster Shapley run over the 41 epsilon values (one row per epsilon, averaged
over the 48 games): MAE, MSE, RMSE, MAPE (denominator true value + 0.1) and the mean
number of unique subsets. This script selects the eight epsilon values shown in Table 2
Panel B and writes them with the computation cost (unique subsets / 255).

Outputs (outputs/main_text/):
  table2_panel_b_from_archived_runs.csv
  table2_panel_b_from_archived_runs.tex

table2_accuracy_efficiency.py regenerates the same panel from the package's own run
(re-collected embeddings); the two agree to within about 0.004 MAE.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / "code" / "algorithms"))
from run_baselines import write_csv  # noqa: E402

ARCHIVE_FILE = PACKAGE / "data" / "archived_runs" / "table2_panel_b" / "metrics_analysis.csv"
PANEL_B_EPSILONS = [0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.01]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive-file", type=Path, default=ARCHIVE_FILE)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "outputs/main_text")
    args = parser.parse_args()

    table = pd.read_csv(args.archive_file)
    rows = []
    for epsilon in PANEL_B_EPSILONS:
        match = table[(table["set_epsilon"] - epsilon).abs() < 1e-9]
        if match.empty:
            raise SystemExit(f"epsilon {epsilon} not in {args.archive_file}")
        r = match.iloc[0]
        rows.append({"epsilon": epsilon, "mae": float(r["mae"]), "mse": float(r["mse"]),
                     "mape": float(r["mape"]), "unique_subsets": float(r["unique_subsets"]),
                     "cost": float(r["unique_subsets"]) / 255.0})

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "table2_panel_b_from_archived_runs.csv", rows)
    lines = ["% Table 2 Panel B from the archived Cluster Shapley run (data/archived_runs/table2_panel_b).",
             "\\begin{tabular}{cccrc}", "\\toprule",
             "\\textbf{Cluster Shapley $\\epsilon$} & \\textbf{MAE} & \\textbf{MSE} & \\textbf{MAPE} & \\textbf{Computation Cost} \\\\",
             "\\midrule"]
    for r in rows:
        lines.append(f"{r['epsilon']:.2f} & {r['mae']:.4f} & {r['mse']:.4f} & {r['mape']:.2f}\\% & {r['cost']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "table2_panel_b_from_archived_runs.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Table 2 Panel B from the archived run")
    print(f"{'epsilon':>8}{'MAE':>9}{'MSE':>9}{'MAPE':>10}{'cost':>7}")
    for r in rows:
        print(f"{r['epsilon']:8.2f}{r['mae']:9.4f}{r['mse']:9.4f}{r['mape']:9.2f}%{r['cost']:7.2f}")
    print(f"Wrote {out / 'table2_panel_b_from_archived_runs.csv'}")


if __name__ == "__main__":
    main()
