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

See the [archived-run data description](DATA_DICTIONARY.md#dataarchived_runs) for those
saved outputs. CASCAD's reproduced Panel B screenshot corresponds to
`table2_panel_b_from_archived_runs.csv`, produced by the unchanged archived-run script.
This is distinct from the regenerated `table2_panel_b.csv`.

## Run Table 2 Panel A independently

```bash
python code/main_text/table2_accuracy_efficiency.py --panel-a-only
```

This needs only the saved games and query/review embeddings, with no preceding algorithm run.
It writes `outputs/main_text/table2_panel_a.csv`, `table2_panel_a.tex` and
`relevance_weights_by_document.csv`. Panel A reports MAE, MSE, MAPE and computation cost.
MAPE is `mean(abs((reference - allocation)/(reference + 0.1))) * 100`, as in Panel B.
The document file retains cosine similarities, allocation
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
coordinates to [-5, 5] for display. Figure 2 reads the separate selected derived input
`data/figure2_example/coalition_scores.csv` and computes its annotations directly.
Its [source record](../data/figure2_example/source_provenance.json) and
[score history](../data/figure2_example/score_history.json) document the score adjustments;
agreement of the sixteen two-decimal annotations does not establish an original LLM-run
identity. Its exact-versus-cluster MAE is approximately 0.057515. Table 1 continues to use
the archived benchmark example game.

Figure 3 pools sampling runs across seeds, games and budgets, then groups them by the exact
number of unique subsets evaluated. The [results map](RESULTS_MAP.md#figure-3-aggregation)
describes the means and intervals used for the plotted curves.

## Reproducibility and optional new experiments

Random draws use fixed seeds, and `run_all.py` runs numerical libraries single-threaded.
Use the same Python version and locked dependencies for repeatable numerical results;
`outputs/logs/run_report.json` records the runtime versions and stage results.

Prompts and optional generation scripts are documented in the
[API generation instructions](#optional-api-regeneration). They collect new summaries, scores or
embeddings and require explicit `--allow-api` and API credentials. New model calls can produce
different summaries or scores; they are separate from offline replication and are never
invoked by `run_all.py`.


## Figure A6 outputs

`figure_a6_lipschitz_check.py` computes all 86,016 document-pair/coalition combinations
from the saved scores and assumption-check distance matrices. The default scatter restores
the original PDF's 4,955 encoded marker placements by selecting existing nonempty-coalition
cache rows using `data/appendix/assumption_check/figure_a6_plot_selection.csv`. The
[recovery record](../data/appendix/assumption_check/figure_a6_plot_recovery.json) contains the
source PDF hash, coordinate transform and matching tolerance. This recovers the geometry,
not every original observation identity: 4,898 placements have multiple matching cache rows.
The remaining 45 points from the stated 5,000 input sample are not invented.

The default `lipschitz_check.pdf` and `.png` use the historical axes
`[0.01, 1] × [0, 10]`, blue points and no added reference line.
`plotted_points.csv` retains the selected source rows and plotting order.
If the archival selection is absent, a fixed-seed sample of 5,000 nonempty-coalition rows
is used; the original random seed is unknown.

A separate `lipschitz_check_full_range.pdf` and `.png` show a fixed-seed diagnostic
over the full distance range. All rows remain in `all_coalitions.csv` and the full-data
statistics. Raw distances are retained; corrected distances clip numerical roundoff to
`[0, 2]`. The corrected local fraction satisfying `difference <= 2.5 * distance` for
distances in `[0, 0.4]` is 86.9246% including empty coalitions, or 87.6367% without
them. This does not reproduce the paper's greater-than-95% statement.
`validation.json` records both distance conventions, plotting frames and counts.

## Individual algorithm runs

These scripts use the included coalition scores and embeddings. Install the
package requirements first. All commands below run offline without notebooks
or model API calls and assume the package directory is the working directory.
Default paths resolve relative to the package; explicit command-line paths
resolve relative to the caller's working directory.

### Baselines

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

### Clustering

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

### Outputs and targets

Baseline outputs default to `outputs/algorithms/baselines/`: `coalition_audit.csv`,
`exact_attributions.csv`, `equal_attribution_by_query.csv`,
`equal_attribution_summary.csv`, `baseline_detailed.csv`, `baseline_summary.csv`
and `run_metadata.json`. Optional outputs are `baseline_comparison.pdf` and
`kernel_numerical_warnings.csv`. The plot uses the stored target and pointwise
95% Student-t intervals over seed-level query means. Clustering outputs default
to `outputs/algorithms/cluster/`: `cluster_assignments.csv`, `cluster_detailed.csv`,
`cluster_summary.csv` and `run_metadata.json`.

The equal-attribution audit files report MAE, MSE and RMSE only. These audit files omit
MAPE; the separate Table 2 Panel A output includes it.

Keep error-table targets separate: `stored` uses input document values;
`recomputed_exact` uses the full Shapley formula with empty-coalition value zero.
Inputs are preserved; invalid or incomplete coalition games raise errors.

### Algorithm settings

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

## Optional API regeneration

These Python scripts create **new stochastic outputs**. They do not recover the missing historical embedding cache or guarantee the paper's original coalition scores. Cached-result reproduction remains the default package workflow. No API calls were made while preparing or testing these scripts.

Python 3.10+ and the standard library suffice for game generation. Embedding export additionally requires NumPy, already used by the offline baseline scripts. There is no notebook dependency, local account configuration, or credential retrieval code.

Every command requires an explicit provider, input and output directory. **Without `--allow-api`, it only validates inputs and prints an upper bound on call count.** It does not read credentials, contact a provider, or write output. Game generation accepts one product-query game per invocation and 1 to 10 retrieved documents. Directory embedding runs default to one CSV unless `--max-products` is increased explicitly. Do not expect a fixed dollar cost: request size and current provider pricing determine charges.

The optional network client expects only the selected `OPENAI_API_KEY` or `OPENROUTER_API_KEY` to be supplied to that process by the operator's approved runtime secret mechanism. Do not put credentials in command arguments, source files, notebooks, manifests, `.env` files or this package. These scripts never print or save credentials or request headers. There is no credential configuration command here. Local project credential policies continue to apply when actually running paid commands.

Run these examples from the package root. Paths containing spaces must remain quoted. The shown commands are dry runs. Append `--allow-api` only after separately authorizing the paid run and supplying its runtime credential.

### Original GPT-4o pipeline

```bash
python code/generation/generate.py original \
  --provider openai \
  --input 'data/coalition_games_8doc/Shapley_the product sucks_gift_card.csv' \
  --output-dir outputs/new_gpt4o_game
```

An archived CSV supplies the retrieved documents and query; its summaries and scores are not reused by this command. Alternatively, pass a JSON file with this schema:

```json
{
  "dataset_name": "Shapley_example.csv",
  "query": "How durable is this product?",
  "product_description": "Optional existing product description",
  "documents": ["First retrieved review", "Second retrieved review"]
}
```

Retrieval is an input, not a newly implemented retrieval system. Document order establishes indices 0 to n-1. The script enumerates every nonempty coalition, labels relevance separately within each coalition, summarizes using `gpt-4o-2024-08-06`, and evaluates all coalition summaries together in each of four independent passes using the same model. All temperatures are 0.1. Summary and evaluation calls use the original strict JSON schemas. The original evaluator is not silently chunked; an oversized request stops with an explicit error. At n=8, the maximum is 514 calls; at n=10, it is 2,050. Wholly irrelevant coalitions do not need a summary-generation call.

CSV loading uses the first row's `original_query`, matching the archived revision loader. One source file, `Shapley_graphics quality of the product_PS5.csv`, has row-level query text that increments PS5 to PS6 and so forth. This anomaly is reported during preflight and recorded in generated metadata; the original CSV is left intact. `generate.py embeddings` uses singleton-summary text only, so its query metadata is recorded for provenance but is not sent as embedding input. JSON inputs specify one query directly.

The CSV output has the original six columns and includes coalition scores plus `Comment i` exact Shapley rows. `generated_game.json` retains generated summaries, subset relevance labels, all four score maps, averaged scores, exact Shapley values, document/query provenance and completion time. The empty coalition is defined to have value zero; missing nonempty coalition values are never replaced with zero.

### Summarizer robustness arm

```bash
python code/generation/generate.py robustness \
  --provider openrouter \
  --input 'data/coalition_games_8doc/Shapley_the product sucks_gift_card.csv' \
  --output-dir outputs/new_summarizer_robustness
```

This requires an archived CSV with every nonempty GPT-4o coalition summary. It preserves the revision experiment's design:

- Re-score cached GPT-4o summaries with the common `google/gemini-2.5-flash` evaluator.
- Label the full retrieved set once with `google/gemini-2.5-flash-lite`, then reuse those relevance labels for all subsets.
- Generate each coalition summary with `google/gemini-2.5-flash-lite`, then evaluate those summaries with the same common Flash evaluator.
- Use four evaluation passes per arm, temperature 0.1, the original OpenRouter backend and `max_tokens=32768`. The revision evaluator chunks only above 300 summaries. With the archived n=8 inputs, all 255 summaries fit one comparative evaluation request per pass.

Outputs are `cached_gpt4o_rescored.csv`, `gemini_regenerated.csv` and `summarizer_robustness.json`. The JSON preserves both arms' four raw score maps and means (`v1`, `v2`), regenerated summaries, full-set relevance labels, exact Shapley values and input provenance. The archived input is not overwritten.

### Singleton-summary embeddings for Cluster Shapley

```bash
python code/generation/generate.py embeddings \
  --provider openai \
  --input data/coalition_games_8doc \
  --max-products 48 \
  --output-dir outputs/new_large_embeddings
```

This embeds the **cached singleton summaries**, not full review text, with `text-embedding-3-large` at 3,072 dimensions. Each numeric `.npy` matrix has shape `(n, 3072)`; row i corresponds exactly to coalition key `str(i)`. The manifest records the exact source CSV basename, query, document order, SHA-256 hashes of the input CSV, individual singleton summary texts and saved array, collection time, requested model, returned model and provider response metadata. A matching model name alone does not establish historical identity.

The output schema is accepted by the offline converter:

```bash
python code/algorithms/prepare_distances.py \
  --embedding-manifest outputs/new_large_embeddings/embedding_manifest.json \
  --output-dir outputs/new_large_distances

python code/algorithms/run_cluster.py \
  --distance-manifest outputs/new_large_distances/distance_manifest.json \
  --output-dir outputs/new_cluster_results
```

The manifest must match every CSV in the selected baseline `--data-dir`. For a bounded one-file experiment, use a separate input directory containing only that CSV and supply the same directory to generation, `prepare_distances.py` and `run_cluster.py`. No credentials are used by the converter or cluster runner.

### Exact prompts, metadata and failure behavior

`prompts.py` preserves six prompt functions separately. Original prompts came from the public `src/llm_summarization.py` and `src/llm_evaluation.py`; revision prompts came from the experiment engine used for the summarizer robustness run. The revision summary prompt omits the original examples about citing every irrelevant comment, and the revision evaluation prompt has shorter steps 1 and 3. Neither version is silently substituted for the other. The historical phrase “We are working on the amazon gift card” is intentionally retained in both relevance prompts for all products. Relevant-only summary inputs, subset-local citation indices, and appended irrelevance statements also retain the source behavior.

`run_config.json` records selected models, temperature, evaluation passes, revision token/chunk limits, prompt-file hash and input hashes. `api_responses.jsonl` records each call's collection time, request hash, nonsecret generation parameters, provider-reported model/usage/response identifiers and raw model output. Thus the raw per-pass responses remain available even if parsing fails. This journal contains research data and should be treated like the other generated data.

There are no automatic paid retries. Incomplete or refused responses, missing relevance labels, duplicate or out-of-range scores, malformed JSON, or missing coalition scores stop the run. This is stricter than the historical revision engine, which could silently average fewer passes or substitute zero for a persistently missing score. Failed results are not presented as completed replications.

To continue after an interruption or transport failure, repeat the **same** command with `--resume --allow-api`. Matching completed calls are reused from the response journal; inputs and prompt configuration must match. A malformed cached response is intentionally not regenerated on resume: keep that directory as an audit record and start a new output directory if you authorize a fresh run. Partial or manually edited journals may fail validation rather than silently losing provenance. The original API model snapshots may be retired; the script does not substitute a newer model automatically.

Offline verification:

```bash
python -m unittest discover -s code/generation -p 'test_*.py' -v
```

Tests use fake model responses only and exercise safe dry runs, exact coalition coverage, four-pass evaluation, checkpoint reuse, malformed scores and embedding export. No actual provider request is made.

### Review and query embeddings for the relevance-weighted rule

```bash
python code/generation/embed_retrieval_inputs.py \
  --input data/coalition_games_8doc \
  --output-dir outputs/new_retrieval_embeddings
```

Embeds, for every archived game, the eight retrieved review strings exactly as stored in
the CSV and the archived query string with `text-embedding-3-large`, and writes
`<game>.reviews.npy`, `<game>.query.npy` and `retrieval_embedding_manifest.json`. Without
`--allow-api` it only validates the inputs. The arrays used by the package were collected
with this script on 2026-09-06 and are stored in `data/embeddings/main_8doc_retrieval/`.


### Two-dimensional embeddings of the Figure 2 example

`embed_figure2_example.py --allow-api` collects the eight two-dimensional embeddings
(`text-embedding-3-large`, `dimensions=2`) of "review title. single-review summary" for the
wireless-controller example and writes `data/embeddings/figure2_example_2d/embeddings_2d.json`.
The stored file is what the Figure 2 scripts use; rerunning the collection is not needed.
