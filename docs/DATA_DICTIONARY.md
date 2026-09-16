# Data dictionary

## Coalition CSVs

`data/coalition_games_8doc/*.csv` and `data/appendix/coalition_games_10doc/*.csv` have the same schema. Each file is one query-product game. Eight-document files contain 255 nonempty coalitions plus eight stored-value rows; ten-document files contain 1,023 plus ten rows.

| Field | Meaning |
| --- | --- |
| `original_query` | Query/attribute used to generate and evaluate the summaries |
| `product_description` | Product description recorded with the game |
| `subset` | A string of document indices, e.g. `013` means documents 0, 1, and 3; `Comment i` denotes a stored document-value row |
| `comments` | Source review texts for the coalition, usually a Python-list literal; use `ast.literal_eval`, never `eval` |
| `summary` | Saved GPT-4o summary for the coalition |
| `avg_score` | Averaged evaluation score for a coalition, or the historical stored document value on a `Comment i` row |

**Read `subset` as text** to preserve leading zeroes. Digit-string encodings support indices 0–9 only. The empty coalition is implicit and has value zero. Historical stored values are preserved as reference data even where they differ from exact recomputation. Filenames are historical labels; document text is the authoritative match for examples with misleading old product names.

## Summarizer robustness JSONs

Each regular filename corresponds to the original CSV stem after removing `Shapley_`.

| Field/file | Meaning |
| --- | --- |
| `scores/*.json: query` | Original query |
| `scores/*.json: docs_n` | Ordered local document indices 0 through 7 |
| `scores/*.json: v1` | Coalition-score dictionary for cached GPT-4o summaries re-evaluated by the common Gemini judge |
| `scores/*.json: v2` | Coalition-score dictionary for Flash-Lite summaries evaluated by the same judge |
| `summaries/*.json` | Dictionary from global coalition keys to generated summary objects (`key`, `summary`) |

Both score dictionaries and every included summary cache have 255 keys. Scores are saved averages; these files do not record individual evaluation passes or response model-version metadata. New generation scripts retain individual passes.

## Assumption-check matrices

`data/appendix/assumption_check/distances/query_*_distance_matrix.npy` contains numeric 8×8 cosine-distance matrices. Load using `numpy.load(..., allow_pickle=False)`. Rows/columns follow document indices 0–7. `distance_matrix_provenance.csv` maps integer query IDs to coalition filenames using a unique full `(subset, summary, score)` signature. Query wording alone is not a unique identifier.

Generating source: singleton-summary embeddings from `text-embedding-3-small`, dimension 256. The main benchmark instead used `text-embedding-3-large`, dimension 3072. These caches must not be silently interchanged.

## Retrieval examples

| Field | Meaning |
| --- | --- |
| `parent_asin` | Product identifier in the source review dataset |
| `helpful_vote` | Recorded helpful-vote count |
| `rating` | Recorded review rating |
| `comment` | Review text used in the selected example |
| `embedding` | Serialized numeric embedding array from the legacy retrieval stage |
| `cosine_similarity` | Saved relevance similarity to the example query |

`sampled_data_with_similarity.csv` contains 1,000 rows and `df_relevant_top30.csv` contains the two 30-review selected sets (60 rows). These are recovered intermediate examples, not the complete main-paper retrieval corpus or the final synthetic-noise results.

## Embeddings (`data/embeddings/`)

Collected on 2026-09-06 with OpenAI `text-embedding-3-large`, 3072 dimensions, float64
NumPy arrays saved with `allow_pickle=False`. Each folder has a JSON manifest that names
the model, the input kind, the SHA-256 of every input text, the provider response ids and
the collection time; `docs/DATA_MANIFEST.csv` carries the file checksums.

| Folder / file | Contents |
| --- | --- |
| `main_8doc_summaries/<game>.embeddings.npy` | 8 x 3072: embeddings of the singleton summaries (coalition rows `0` to `7`) of the game, in document order; the clustering input of Cluster Shapley |
| `main_8doc_summaries/embedding_manifest.json` | Manifest; `files` maps each game CSV name to its array |
| `ten_doc_summaries/` | Same for the two ten-document games (10 x 3072) |
| `main_8doc_retrieval/<game>.reviews.npy` | 8 x 3072: embeddings of the archived review strings (`title: ...\ntext: ...`) of the game; used for the relevance-weighted rule |
| `main_8doc_retrieval/<game>.query.npy` | 3072: embedding of the game's archived query string |
| `main_8doc_retrieval/retrieval_embedding_manifest.json` | Manifest; `files` maps each game CSV name to its two arrays |

The original embedding cache of the research notebooks was not archived; these arrays are
a new collection and are labelled as such in every manifest.


## `data/archived_runs/`

Saved outputs of the original notebook runs behind Figure 3 and Table 2 Panel B (Monte Carlo,
Truncated Monte Carlo and Kernel SHAP per seed and game; adaptive Cluster Shapley per game and
epsilon; the Cluster Shapley summary per epsilon). Columns are documented in
`data/archived_runs/README.md`.

## `data/embeddings/figure2_example_2d/`

`embeddings_2d.json`: the eight two-dimensional embeddings behind Figure 2 (`text-embedding-3-large`
with `dimensions=2`, input "review title. single-review summary", texts included in the file),
collected on 2026-09-15 with `code/generation/embed_figure2_example.py`; `api_responses.jsonl` is
the credential-free response journal of that call.
