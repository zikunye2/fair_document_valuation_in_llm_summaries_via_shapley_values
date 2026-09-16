# Fair Document Valuation in LLM Summaries via Shapley Values

Code and data accompanying [Ye and Yoganarasimhan (2026)](https://arxiv.org/abs/2505.23842), accepted at *Management Science*.

This project studies how to assign credit to the documents used in an LLM-generated summary. It implements exact Shapley, Monte Carlo, Truncated Monte Carlo, Kernel SHAP and Cluster Shapley, with data and scripts for the paper's tables and figures.

The default workflow runs offline using the provided model-generated summaries, evaluation scores and embeddings. No API key or Jupyter is required.

## Quick start

From the repository directory, using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python run_all.py
```

Results and logs are written to `outputs/`. See the [replication guide](docs/REPLICATION_GUIDE.md) for setup on Windows, individual runs, output locations and calculation details.

## Repository structure

| Path | Contents |
| --- | --- |
| [run_all.py](run_all.py) | Entry point for the full offline workflow |
| [code/algorithms/](code/algorithms/) | Shapley valuation algorithms and algorithm checks |
| [code/main_text/](code/main_text/) | Scripts for the main-text tables and figures |
| [code/appendix/](code/appendix/) | Scripts for the Web Appendix results |
| [code/generation/](code/generation/) | Optional scripts for collecting new model outputs |
| [data/](data/) | Scored coalition games, embeddings, archived experiment outputs and appendix inputs |
| [docs/](docs/) | Replication instructions, result-to-script map and data documentation |

## Documentation

- [Replication guide](docs/REPLICATION_GUIDE.md): installation, execution and how to read the outputs.
- [Results map](docs/RESULTS_MAP.md): the scripts, inputs and outputs for each table and figure.
- [Data dictionary](docs/DATA_DICTIONARY.md): datasets, file formats and fields.

## Reference

Ye, Zikun, and Hema Yoganarasimhan. 2026. *Fair Document Valuation in LLM Summaries via Shapley Values*. Management Science, forthcoming. [Paper](https://arxiv.org/abs/2505.23842).
