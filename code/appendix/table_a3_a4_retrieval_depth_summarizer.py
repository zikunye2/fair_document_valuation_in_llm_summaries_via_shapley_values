#!/usr/bin/env python3
"""Reproduce revision tables from saved coalition games, without any API calls."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import kendalltau, spearmanr

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'code' / 'algorithms'))
from run_baselines import load_game, write_csv


def legacy_exact(scores, docs):
    """Original revision script's accumulation order and normalization.

    Completeness is checked before computation; a missing coalition is never
    replaced by zero. Digit-string IDs describe at most ten documents.
    """
    n = len(docs)
    allowed = set(map(str, docs))
    items = [(k, float(v)) for k, v in scores.items() if k and set(k) <= allowed]
    masks = [frozenset(map(int, k)) for k, _ in items]
    if len(set(masks)) != 2**n - 1 or len(masks) != len(set(masks)):
        raise ValueError('Incomplete or duplicate coalition game')
    if any(len(k) != len(set(k)) or not math.isfinite(v) for k, v in items):
        raise ValueError('Malformed coalition or nonfinite score')
    cache, phi = {frozenset(): 0.0}, dict.fromkeys(docs, 0.0)
    for key, value in sorted(items, key=lambda item: len(item[0])):
        subset = frozenset(map(int, key))
        cache[subset] = value
        weight = math.factorial(len(subset)-1) * math.factorial(n-len(subset)) / math.factorial(n)
        for i in map(int, key):
            phi[i] += (value - cache[subset - {i}]) * weight
    full, total = cache[frozenset(docs)], sum(phi.values())
    return np.array([phi[i] / total * full if total else phi[i] for i in docs])


def read_scores(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    return {r['subset']: float(r['avg_score']) for r in rows if r['subset'].isdigit()}


def rank_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError('All rank correlations were undefined')
    return dict(mean=float(np.mean(values)), median=float(np.median(values)), n=int(len(values)))


def main():
    output = PACKAGE / 'outputs' / 'appendix' / 'table_a3_a4'
    output.mkdir(parents=True, exist_ok=True)
    files = sorted((PACKAGE / 'data' / 'coalition_games_8doc').glob('*.csv'))
    if len(files) != 48:
        raise ValueError('Expected all 48 query-product coalition files')
    retrieval, summarizer, attributions = [], [], []
    for path in files:
        load_game(path)  # Validate the complete original score and target schema.
        scores = read_scores(path)
        phis = {k: legacy_exact(scores, list(range(k))) for k in (4, 6, 8)}
        for k in (4, 6):
            constant = np.ptp(phis[k]) == 0 or np.ptp(phis[8][:k]) == 0
            retrieval.append(dict(file=path.name, k=k,
                spearman=float('nan') if constant else float(spearmanr(phis[k], phis[8][:k]).statistic),
                kendall=float('nan') if constant else float(kendalltau(phis[k], phis[8][:k]).statistic)))
        for k, phi in phis.items():
            for i, value in enumerate(phi):
                attributions.append(dict(file=path.name, k=k, document=i, shapley=float(value)))
        cache_file = PACKAGE / 'data' / 'appendix' / 'revision' / 'scores' / (path.stem.removeprefix('Shapley_') + '.json')
        cache = json.loads(cache_file.read_text())
        if cache['docs_n'] != list(range(8)):
            raise ValueError(f'{cache_file.name}: unexpected document IDs')
        p1 = legacy_exact(cache['v1'], list(range(8)))
        p2 = legacy_exact(cache['v2'], list(range(8)))
        summaries = json.loads((PACKAGE / 'data' / 'appendix' / 'revision' / 'summaries' / cache_file.name).read_text())
        if set(summaries) != set(cache['v1']) or set(summaries) != set(cache['v2']):
            raise ValueError(f'{cache_file.name}: summary and score keys differ')
        summarizer.append(dict(file=path.name, mad=float(np.mean(np.abs(p1-p2))),
            gpt4o_within_query_range=float(np.ptp(p1)),
            **{f'gpt4o_doc{i}': float(x) for i,x in enumerate(p1)},
            **{f'flash_lite_doc{i}': float(x) for i,x in enumerate(p2)}))
    write_csv(output / 'retrieval_per_query.csv', retrieval)
    write_csv(output / 'retrieval_shapley_values.csv', attributions)
    write_csv(output / 'summarizer_per_query.csv', summarizer)
    retrieval_table = []
    for k in (4, 6):
        row = dict(comparison=f'K={k} vs K=8')
        for metric in ('spearman', 'kendall'):
            for key, value in rank_summary([r[metric] for r in retrieval if r['k'] == k]).items():
                row[f'{metric}_{key}'] = value
        retrieval_table.append(row)
    mad = np.array([r['mad'] for r in summarizer])
    summarizer_table = dict(n=len(mad), median=float(np.median(mad)), mean=float(mad.mean()),
        q25=float(np.quantile(mad, .25)), q75=float(np.quantile(mad, .75)))
    write_csv(output / 'table_vary_k.csv', retrieval_table)
    write_csv(output / 'table_vary_summarizer.csv', [summarizer_table])
    checks = dict(query_product_pairs=48,
        complete_coalition_games=True,
        summary_score_keys_matched=True,
        median_gpt4o_within_query_range=float(np.median([r['gpt4o_within_query_range'] for r in summarizer])),
        evaluation_pass_provenance='Caches contain averaged scores, not individual scoring passes.')
    (output / 'validation.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
