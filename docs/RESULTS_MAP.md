# Results map: which script produces which result

For installation and step-by-step execution, start with the [replication guide](REPLICATION_GUIDE.md).

`python run_all.py` runs the stages below in this order and writes everything under `outputs/`.
Paths are relative to the package directory. A script can be run on its own once the stages it
depends on have run (`python <script> --help` lists its options).

## Stages

| Stage | Script | Inputs | Outputs |
| --- | --- | --- | --- |
| check_data | `code/verify_inventory.py` | `docs/DATA_MANIFEST.csv`, `data/` | (checksum check only) |
| test_algorithms | `code/algorithms/test_algorithms.py` | none | (unit tests) |
| distances | `code/algorithms/prepare_distances.py` | `data/embeddings/main_8doc_summaries/` | `outputs/algorithms/distances/` (cosine-distance matrix per game) |
| cluster_shapley | `code/algorithms/run_cluster.py` | `data/coalition_games_8doc/`, `outputs/algorithms/distances/` | `outputs/algorithms/cluster/` (`cluster_detailed.csv`, `cluster_assignments.csv`, `cluster_summary.csv`; adaptive and standard DBSCAN, 41 epsilon values) |
| baselines | `code/algorithms/run_baselines.py --methods mc tmc kernel` | `data/coalition_games_8doc/` | `outputs/algorithms/baselines/` (`baseline_detailed.csv`, `baseline_summary.csv`, audits, `run_metadata.json`) |
| table1 | `code/main_text/table1_example_shapley.py` | the wireless-controller game | `outputs/main_text/table1_example_shapley.csv` |
| figure2 | `code/main_text/figure2_clustering_visualization.py` | the archived example game, `data/embeddings/figure2_example_2d/embeddings_2d.json` | `outputs/main_text/figure2_clustering_visualization.pdf`, `.png`, `_points.csv` |
| figure3 | `code/main_text/figure3_benchmark_comparison.py` | `outputs/algorithms/baselines/`, `outputs/algorithms/cluster/` | `outputs/main_text/figure3_benchmark_comparison.pdf`, `.png`, `_points.csv` |
| table2 | `code/main_text/table2_accuracy_efficiency.py` | `data/coalition_games_8doc/`, `outputs/algorithms/cluster/`, `data/embeddings/main_8doc_retrieval/` | `outputs/main_text/table2_panel_a.csv`, `.tex`, `table2_panel_b.csv`, `.tex`, `table2_accuracy_efficiency.tex`, `relevance_weights_by_document.csv` |
| figure3_archived_runs | `code/main_text/figure3_from_archived_runs.py` | `data/archived_runs/figure3_baselines/`, `data/archived_runs/figure3_cluster/` | `outputs/main_text/figure3_from_archived_runs.pdf`, `.png`, `_points.csv` |
| table2_panel_b_archived_runs | `code/main_text/table2_panel_b_from_archived_runs.py` | `data/archived_runs/table2_panel_b/metrics_analysis.csv` | `outputs/main_text/table2_panel_b_from_archived_runs.csv`, `.tex` |
| appendix_audit_8doc, appendix_audit_10doc | `code/algorithms/run_baselines.py --audit-only` | `data/coalition_games_8doc/`, `data/appendix/coalition_games_10doc/` | `outputs/appendix/audit_8doc/`, `outputs/appendix/audit_10doc/` |
| appendix_table_a3_a4 | `code/appendix/table_a3_a4_retrieval_depth_summarizer.py` | `data/coalition_games_8doc/`, `data/appendix/revision/` | `outputs/appendix/table_a3_a4/` |
| appendix_figure_a6 | `code/appendix/figure_a6_lipschitz_check.py` | `data/coalition_games_8doc/`, `data/appendix/assumption_check/`, `docs/distance_matrix_provenance.csv` | `outputs/appendix/figure_a6_lipschitz/` |

`--skip-appendix` leaves out the last four stages; `--quick-baselines` shrinks the baselines
stage to a smoke test (two games, two seeds, three budgets), after which the regenerated Figure 3
is not the paper figure.

Panel A can also run directly from the saved coalition scores and query/review embeddings:
`python code/main_text/table2_accuracy_efficiency.py --panel-a-only`. It needs no preceding
stage or API call and writes `outputs/main_text/table2_panel_a.csv`, `.tex`, and
`relevance_weights_by_document.csv`. Panel A computes MAE and MSE, and reports computation
cost; it does not calculate percentage errors. Panel B retains MAE, MSE, MAPE and cost.
The default Table 2 stage writes both panel CSVs and TeX tables, plus a combined TeX file
with each panel's own columns. The retired combined CSV is removed from the output directory
on rerun so it cannot retain old Panel A percentage-error results.

## Targets

`stored` refers to the Shapley values saved in each game's `Comment i` rows. `recomputed_exact`
is calculated from the coalition scores with the standard Shapley weights and the empty coalition
at value zero. The two can differ slightly. Baseline and cluster outputs carry both targets;
archived-run scripts preserve the error metrics as reported in those runs. Table 1 reports both.

## Figure 3 aggregation

For each sampling method, the runs are pooled over seeds, games and budgets and grouped by the
exact number of unique subsets evaluated; the curve is the mean MAE of each group and the band
its 95% t interval, as in the research notebook that drew the published figure. Grouping by
budget instead (mean unique subsets and mean MAE per number of permutations) gives a different
Truncated Monte Carlo curve, because games that truncate heavily never reach large subset counts.
Cluster Shapley points are means over the 48 games at each epsilon.

## What to expect

- The archived-run Figure 3 and Panel B scripts read saved experiment outputs. The regenerated
  versions instead compute allocations and errors from fixed coalition scores and embeddings.
  No prompt is rerun. In regenerated Panel B all metrics come from the same `separated`
  (adaptive Cluster Shapley) run and the same reference target. The adaptive procedure shrinks
  the distance threshold until each cluster satisfies the within-cluster pairwise distance rule.
- Figure 2 positions and clusters are the published ones: two-dimensional API embeddings of the
  eight reviews (stored in `data/embeddings/figure2_example_2d/`), DBSCAN with eps 0.05 on their
  cosine distances, min-max scaling for display.
- Table 1 and the standard Figure 2 script calculate values from the archived example game.
- Tables A3 and A4 and the Figure A6 data are computed from the saved caches.

Every random draw uses a fixed seed and `run_all.py` runs the numerical libraries
single-threaded, so repeated runs in the same environment give identical outputs.
