# Replication guide

Run all commands below from the package directory, which contains `run_all.py`.
The workflow uses saved inputs and runs offline; it needs neither an API key nor Jupyter.
See the [results map](RESULTS_MAP.md) for scripts, dependencies and outputs, and the
[data dictionary](DATA_DICTIONARY.md) for file formats and fields.

## Install

Python 3.12 is recommended. Python 3.14 has also been tested.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

On Windows (Command Prompt), use `py -3.12 -m venv .venv` and `.venv\Scripts\activate.bat`.
For Python 3.14, use `python3.14` in the first command above.

## Run the workflow

```bash
python run_all.py
```

The full workflow takes about two minutes on a laptop. It first checks the data against
`docs/DATA_MANIFEST.csv`, then runs the algorithm tests, main-text stages and appendix stages.

| Command | Scope |
| --- | --- |
| `python run_all.py` | Complete offline workflow |
| `python run_all.py --skip-appendix` | Data checks, tests and main-text stages |
| `python run_all.py --quick-baselines` | Workflow with a smaller sampling-benchmark run: two games, two seeds and three budgets |

The quick option is a smoke test; its regenerated Figure 3 uses the reduced benchmark sample.
It can be combined with `--skip-appendix`.

Results are written to `outputs/main_text/` and `outputs/appendix/`; intermediate algorithm
outputs are in `outputs/algorithms/`. Stage logs and `run_report.json` are in `outputs/logs/`.
If a stage stops, its log gives the error. Every script accepts `--help`; the
[results map](RESULTS_MAP.md) lists the prerequisites for running a stage independently.

## Saved inputs and the two result workflows

The review data cover 48 query-product pairs from [Amazon Reviews 2023](https://arxiv.org/abs/2403.03952).
Only the retrieved reviews are included. Each eight-document game in `data/coalition_games_8doc/`
stores a GPT-4o summary and score for all 255 nonempty coalitions, plus the saved document values.
Scores range from 0 to 10 and average four GPT-4o evaluations. Embeddings are saved under
`data/embeddings/`. Calculations use these fixed inputs without new model calls or score changes.

The package provides two ways to produce Figure 3 and Table 2 Panel B:

- The standard algorithm stages calculate allocations and error metrics from fixed games and embeddings.
- The archived-run scripts redraw curves and format tables from previously computed notebook
  results in `data/archived_runs/`.

Both are included in the full workflow. To run just the archived-run scripts:

```bash
python code/main_text/figure3_from_archived_runs.py
python code/main_text/table2_panel_b_from_archived_runs.py
```

See the [archived-run description](../data/archived_runs/README.md) for those saved outputs.

## Run Table 2 Panel A independently

```bash
python code/main_text/table2_accuracy_efficiency.py --panel-a-only
```

This needs only the saved games and query/review embeddings, with no preceding algorithm run.
It writes `outputs/main_text/table2_panel_a.csv`, `table2_panel_a.tex` and
`relevance_weights_by_document.csv`. Panel A reports MAE, MSE and computation cost; it does
not calculate percentage errors. The document file retains cosine similarities, allocation
weights, equal and relevance-weighted allocations, and reference Shapley values.

The default Table 2 stage produces both panels, each with its own CSV and TeX file, plus
`table2_accuracy_efficiency.tex` containing both panels. Panel B reports MAE, MSE, MAPE and
cost from the same adaptive Cluster Shapley predictions and selected reference target.

## Reference targets and figure settings

`stored` means the document values saved in each game's `Comment i` rows. `recomputed_exact`
means exact Shapley values calculated from the saved coalition scores with the empty coalition
at value zero. The panel CSVs include both targets. Regenerated Figure 3 and Table 2 TeX use
`stored` by default; Figure 3 also accepts `--target recomputed_exact`. Archived-run scripts
retain the metrics recorded in their input files.

Figure 2 uses saved two-dimensional `text-embedding-3-large` vectors of each review's title
and single-review summary. They were requested directly with `dimensions=2`. The offline
script applies DBSCAN (`eps=0.05`, `min_samples=1`) to cosine distances and min-max scales the
coordinates to [-5, 5] for display. Table 1 and Figure 2 values come from the archived game.

Figure 3 pools sampling runs across seeds, games and budgets, then groups them by the exact
number of unique subsets evaluated. The [results map](RESULTS_MAP.md#figure-3-aggregation)
describes the means and intervals used for the plotted curves.

## Reproducibility and optional new experiments

Random draws use fixed seeds, and `run_all.py` runs numerical libraries single-threaded.
Use the same Python version and locked dependencies for repeatable numerical results;
`outputs/logs/run_report.json` records the runtime versions and stage results.

Prompts and optional generation scripts are documented in the
[API generation guide](../code/generation/README.md). They collect new summaries, scores or
embeddings and require explicit `--allow-api` and API credentials. New model calls can produce
different summaries or scores; they are separate from offline replication and are never
invoked by `run_all.py`.
