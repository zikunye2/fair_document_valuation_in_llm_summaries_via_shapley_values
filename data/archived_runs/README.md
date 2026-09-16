# Archived notebook runs behind Figure 3 and Table 2 Panel B

These files are the saved outputs of the original research runs on the 48 coalition games in
`data/coalition_games_8doc/`, as run in the research notebooks in 2025. They are included so
that Figure 3 curves and Table 2 Panel B rows can be reconstructed from these saved outputs.
The package also computes fresh allocations and errors from the fixed coalition scores and
embeddings (`code/main_text/figure3_benchmark_comparison.py`,
`code/main_text/table2_accuracy_efficiency.py`). The archived-run scripts preserve the errors
reported in the saved files; they do not recompute Shapley values or error metrics.

| Folder | Contents |
| --- | --- |
| `figure3_baselines/monte_carlo/` | `permutation_detailed_results_{0-9}_mape_0.1.csv`: Monte Carlo permutation sampling, seeds 0 to 9 |
| `figure3_baselines/truncated_monte_carlo/` | `tmc_0.5_detailed_results_{0-9}_mape_0.1.csv`: Truncated Monte Carlo, performance tolerance 0.5, seeds 0 to 9 |
| `figure3_baselines/kernel_shap/` | `kernel_shap_detailed_results_{0-9}_mape_0.1.csv`: Kernel SHAP, seeds 0 to 9 |
| `figure3_cluster/` | `clustering_results_with_step.csv`: adaptive Cluster Shapley, one row per game and epsilon (41 epsilon values) |
| `table2_panel_b/` | `metrics_analysis.csv`: the Cluster Shapley summary over the 48 games, one row per epsilon |

Columns of the baseline files: `iterations` (number of permutations, or Kernel SHAP samples),
`dataset_id` (game index in the notebook's file order, 0 to 47), `mae`, `mse`, `rmse`, `mape`
(denominator true value + 0.1), `unique_subsets` (distinct coalitions evaluated); the Truncated
Monte Carlo and Kernel SHAP files also have `total_subsets` (evaluations including repeats) and
the Truncated Monte Carlo files `tolerance` (0.5). Errors are against the exact Shapley values
of each game.

Columns of `clustering_results_with_step.csv`: `iterations` (the epsilon value), `dataset_id`,
`mae`, `mse`, `rmse`, `unique_subsets`, `num_clusters`, `actual_epsilon` (after the adaptive
shrinking), `step` (number of shrinking steps). `metrics_analysis.csv` averages these over the
games and adds `mape` and `sparsity` (share of the 255 coalitions not evaluated).

Provenance: delivered by Yizhuo Chang on 2026-09-14 (`organized.zip`), copied here unchanged.
