# Web Appendix scripts

Run by `python run_all.py` unless `--skip-appendix` is given; each script can also be run on
its own from the package directory.

| Script | Result | Inputs | Outputs |
| --- | --- | --- | --- |
| `table_a3_a4_retrieval_depth_summarizer.py` | Table A3 (robustness to retrieval depth K) and Table A4 (robustness to the summarizer model) | `data/coalition_games_8doc/`, `data/appendix/revision/` (coalition scores and summaries of the two summarizer arms) | `outputs/appendix/table_a3_a4/` |
| `figure_a6_lipschitz_check.py` | Figure A6 (empirical check of the Lipschitz continuity assumption) | `data/coalition_games_8doc/`, `data/appendix/assumption_check/` (cosine-distance matrices), `docs/distance_matrix_provenance.csv` | `outputs/appendix/figure_a6_lipschitz/` |
| `../algorithms/run_baselines.py --audit-only` | consistency audit of the archived coalition games (8-document and 10-document) | `data/coalition_games_8doc/`, `data/appendix/coalition_games_10doc/` | `outputs/appendix/audit_8doc/`, `outputs/appendix/audit_10doc/` |

`data/appendix/retrieval_examples/` holds the retrieved-review samples of the two products used
in the Web Appendix's large-document experiment; that experiment is not rerun here.
