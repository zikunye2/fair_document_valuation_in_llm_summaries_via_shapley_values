# Fair Document Valuation in LLM Summaries via Shapley Values

Code and data for [Ye and Yoganarasimhan (2026)](https://arxiv.org/abs/2505.23842), accepted at *Management Science*.

Run the supplied scores, summaries and embeddings offline. No API key or Jupyter is required.

## Install and reproduce

From this directory, using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python run_all.py
```

Results are written to `outputs/main_text/` and `outputs/appendix/`; intermediate
calculations are in `outputs/algorithms/` and logs in `outputs/logs/`.
Use `--quick-baselines` for a smoke test or `--skip-appendix` for main-text stages only.
The [replication guide](docs/REPLICATION_GUIDE.md) covers Windows setup and individual runs.

## Table 2 outputs

Panel A can run independently:

```bash
python code/main_text/table2_accuracy_efficiency.py --panel-a-only
```

The Panel B output displayed and accepted in CASCAD's report is the archived-run version:

```bash
python code/main_text/table2_panel_b_from_archived_runs.py
```

It writes `outputs/main_text/table2_panel_b_from_archived_runs.csv` and `.tex`.
The full workflow retains both archived and regenerated result routes; their outputs have
different filenames. Panel A includes MAE, MSE and stabilized MAPE.

## Figure inputs

Figure 2 reads a selected derived [score file](data/figure2_example/coalition_scores.csv).
It calculates annotations directly from that file. The [source record](data/figure2_example/source_provenance.json)
and [generation record](data/figure2_example/score_history.json) document its derivation.

Figure A6 restores the archived marker selection from existing cache rows. Its
[recovery record](data/appendix/assumption_check/figure_a6_plot_recovery.json) and
[guide](docs/REPLICATION_GUIDE.md#figure-a6-outputs) describe the plotting window,
full-range diagnostic and verification limits.

See the [results map](docs/RESULTS_MAP.md) for input/output paths and the
[data dictionary](docs/DATA_DICTIONARY.md) for file formats. Optional new model calls
are documented in the [generation instructions](docs/REPLICATION_GUIDE.md#optional-api-regeneration)
and are separate from offline reproduction.
