# Offline baseline and clustering analyses

These scripts use the included coalition scores and embeddings. Install the
package requirements first. All commands below run offline without notebooks
or model API calls and assume the package directory is the working directory.
Default paths resolve relative to the package; explicit command-line paths
resolve relative to the caller's working directory.

## Baselines

```bash
python code/algorithms/run_baselines.py --methods mc tmc kernel --plot
python code/algorithms/test_algorithms.py
```

The full run uses 48 games in `data/coalition_games_8doc/`, ten seeds (0–9), and the
method-specific budget grids in `run_baselines.py`. Default methods are MC and
TMC. For two games, two seeds and budgets 5, 20 and 100, use a separate smoke output:

```bash
python code/algorithms/run_baselines.py --quick --methods mc tmc kernel --plot --output-dir outputs/algorithms/baselines_smoke
python code/algorithms/run_baselines.py --audit-only --data-dir data/appendix/coalition_games_10doc --output-dir outputs/appendix/audit_10doc
```

`--audit-only` computes input checks, exact and equal attributions without
sampling. MC, TMC and clustering require NumPy. Kernel SHAP additionally needs
SHAP and its dependencies; plotting requires Matplotlib and SciPy.

## Clustering

```bash
python code/algorithms/prepare_distances.py --embedding-manifest data/embeddings/main_8doc_summaries/embedding_manifest.json
python code/algorithms/run_cluster.py --distance-manifest outputs/algorithms/distances/distance_manifest.json
```

The grid has 41 epsilon values: `0.01`, followed by `i/40` for `i = 1, ..., 40`.
Override it with `--epsilons`. Add `--cluster-permutations 10 20 100` to also run
cluster-level MC (ten seeds by default); use a separate `--output-dir` for this run.
For ten-document clustering, use `data/embeddings/ten_doc_summaries/embedding_manifest.json`,
pass `--data-dir data/appendix/coalition_games_10doc` to both commands, and choose separate
output directories, passing the newly prepared distance manifest to the second command.

Manifest `files` paths are relative to the manifest. Embedding arrays have shape
`n_documents × dimensions`; distance arrays have shape `n_documents × n_documents`.
Row/column `i` corresponds to document `i`; arrays load with `allow_pickle=False`.
The default input specification is singleton summaries embedded with
`text-embedding-3-large`, dimension 3072. Alternative models require
`--allow-alternative-model`. Small-model assumption-check inputs serve a separate analysis.

## Outputs and targets

Baseline outputs default to `outputs/algorithms/baselines/`: `coalition_audit.csv`,
`exact_attributions.csv`, `equal_attribution_by_query.csv`,
`equal_attribution_summary.csv`, `baseline_detailed.csv`, `baseline_summary.csv`
and `run_metadata.json`. Optional outputs are `baseline_comparison.pdf` and
`kernel_numerical_warnings.csv`. The plot uses the stored target and pointwise
95% Student-t intervals over seed-level query means. Clustering outputs default
to `outputs/algorithms/cluster/`: `cluster_assignments.csv`, `cluster_detailed.csv`,
`cluster_summary.csv` and `run_metadata.json`.

The equal-attribution audit files report MAE, MSE and RMSE only. Percentage errors are
computed for the sampling and clustering benchmark methods, not for equal attribution.

Keep error-table targets separate: `stored` uses input document values;
`recomputed_exact` uses the full Shapley formula with empty-coalition value zero.
Inputs are preserved; invalid or incomplete coalition games raise errors.

## Algorithm settings

MC samples permutations with replacement using Python `random.shuffle`, averages
marginal contributions and normalizes to the full-coalition score. TMC uses
NumPy `RandomState` permutations and tolerance `0.5`; `best_score_seen` persists
across permutations, and a coalition is evaluated when that best score minus
the current score is at least `0.5` (or no best score has been recorded).
Kernel SHAP uses identity link, `l1_reg="aic"`, a zero background and all-ones input.
Stabilized MAPE is `mean(abs((target-estimate)/(target+0.1)))*100`.
MC/TMC costs count distinct nonempty coalitions; Kernel SHAP also counts the empty
coalition. Baseline summaries average documents, queries, then seed-level means.
DBSCAN uses `min_samples=1`. Separated clustering shrinks epsilon by `0.95` until
same-cluster pairs coincide with pairs meeting the distance rule (100-iteration limit).
Both compute exact cluster-game Shapley values and divide them equally within clusters.
Optional cluster MC uses a fresh `random.Random(seed)` per query, epsilon and budget.
