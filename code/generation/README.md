# Optional API regeneration

These Python scripts create **new stochastic outputs**. They do not recover the missing historical embedding cache or guarantee the paper's original coalition scores. Cached-result reproduction remains the default package workflow. No API calls were made while preparing or testing these scripts.

Python 3.10+ and the standard library suffice for game generation. Embedding export additionally requires NumPy, already used by the offline baseline scripts. There is no notebook dependency, local account configuration, or credential retrieval code.

Every command requires an explicit provider, input and output directory. **Without `--allow-api`, it only validates inputs and prints an upper bound on call count.** It does not read credentials, contact a provider, or write output. Game generation accepts one product-query game per invocation and 1 to 10 retrieved documents. Directory embedding runs default to one CSV unless `--max-products` is increased explicitly. Do not expect a fixed dollar cost: request size and current provider pricing determine charges.

The optional network client expects only the selected `OPENAI_API_KEY` or `OPENROUTER_API_KEY` to be supplied to that process by the operator's approved runtime secret mechanism. Do not put credentials in command arguments, source files, notebooks, manifests, `.env` files or this package. These scripts never print or save credentials or request headers. There is no credential configuration command here. Local project credential policies continue to apply when actually running paid commands.

Run these examples from the package root. Paths containing spaces must remain quoted. The shown commands are dry runs. Append `--allow-api` only after separately authorizing the paid run and supplying its runtime credential.

## Original GPT-4o pipeline

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

CSV loading uses the first row's `original_query`, matching the archived revision loader. One source file, `Shapley_graphics quality of the product_PS5.csv`, has row-level query text that increments PS5 to PS6 and so forth. This anomaly is reported during preflight and recorded in generated metadata; the original CSV is left intact. Embedding requests use singleton-summary text only, so query metadata is recorded for provenance but is not sent as embedding input. JSON inputs specify one query directly.

The CSV output has the original six columns and includes coalition scores plus `Comment i` exact Shapley rows. `generated_game.json` retains generated summaries, subset relevance labels, all four score maps, averaged scores, exact Shapley values, document/query provenance and completion time. The empty coalition is defined to have value zero; missing nonempty coalition values are never replaced with zero.

## Summarizer robustness arm

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

## Singleton-summary embeddings for Cluster Shapley

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

## Exact prompts, metadata and failure behavior

`prompts.py` preserves six prompt functions separately. Original prompts came from the public `src/llm_summarization.py` and `src/llm_evaluation.py`; revision prompts came from the experiment engine used for the summarizer robustness run. The revision summary prompt omits the original examples about citing every irrelevant comment, and the revision evaluation prompt has shorter steps 1 and 3. Neither version is silently substituted for the other. The historical phrase “We are working on the amazon gift card” is intentionally retained in both relevance prompts for all products. Relevant-only summary inputs, subset-local citation indices, and appended irrelevance statements also retain the source behavior.

`run_config.json` records selected models, temperature, evaluation passes, revision token/chunk limits, prompt-file hash and input hashes. `api_responses.jsonl` records each call's collection time, request hash, nonsecret generation parameters, provider-reported model/usage/response identifiers and raw model output. Thus the raw per-pass responses remain available even if parsing fails. This journal contains research data and should be treated like the other generated data.

There are no automatic paid retries. Incomplete or refused responses, missing relevance labels, duplicate or out-of-range scores, malformed JSON, or missing coalition scores stop the run. This is stricter than the historical revision engine, which could silently average fewer passes or substitute zero for a persistently missing score. Failed results are not presented as completed replications.

To continue after an interruption or transport failure, repeat the **same** command with `--resume --allow-api`. Matching completed calls are reused from the response journal; inputs and prompt configuration must match. A malformed cached response is intentionally not regenerated on resume: keep that directory as an audit record and start a new output directory if you authorize a fresh run. Partial or manually edited journals may fail validation rather than silently losing provenance. The original API model snapshots may be retired; the script does not substitute a newer model automatically.

Offline verification:

```bash
python -m unittest discover -s code/generation -p 'test_*.py' -v
```

Tests use fake model responses only and exercise safe dry runs, exact coalition coverage, four-pass evaluation, checkpoint reuse, malformed scores and embedding export. No actual provider request is made.

## Review and query embeddings for the relevance-weighted rule

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


## Two-dimensional embeddings of the Figure 2 example

`embed_figure2_example.py --allow-api` collects the eight two-dimensional embeddings
(`text-embedding-3-large`, `dimensions=2`) of "review title. single-review summary" for the
wireless-controller example and writes `data/embeddings/figure2_example_2d/embeddings_2d.json`.
The stored file is what the Figure 2 scripts use; rerunning the collection is not needed.
