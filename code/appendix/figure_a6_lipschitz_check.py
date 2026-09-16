#!/usr/bin/env python3
"""Reconstruct the assumption check with the correctly identified small-model cache."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'code' / 'algorithms'))
from run_baselines import load_game, write_csv


def main():
    out = PACKAGE / 'outputs' / 'appendix' / 'figure_a6_lipschitz'
    out.mkdir(parents=True, exist_ok=True)
    with (PACKAGE / 'docs' / 'distance_matrix_provenance.csv').open() as stream:
        mapping = list(csv.DictReader(stream))
    if (len(mapping) != 48
        or {int(row['legacy_query_id']) for row in mapping} != set(range(48))
        or len({Path(row['original_coalition_csv']).name for row in mapping}) != 48
        or len({Path(row['original_distance_matrix']).name for row in mapping}) != 48):
        raise ValueError('Expected a unique mapping for 48 query-product games')
    rows = []
    for item in mapping:
        query = int(item['legacy_query_id'])
        values, _ = load_game(PACKAGE / 'data' / 'coalition_games_8doc' / Path(item['original_coalition_csv']).name)
        matrix = np.load(PACKAGE / 'data' / 'appendix' / 'assumption_check' / 'distances' / Path(item['original_distance_matrix']).name, allow_pickle=False)
        if matrix.shape != (8, 8) or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
            raise ValueError(f'Invalid distance matrix for query {query}')
        for i in range(8):
            for j in range(i+1,8):
                for mask in range(256):
                    if mask & ((1 << i) | (1 << j)):
                        continue
                    rows.append(dict(query_id=query, doc_i=i, doc_j=j, coalition_mask=mask,
                        coalition_size=mask.bit_count(), embedding_distance=float(matrix[i,j]),
                        marginal_contribution_diff=float(abs(values[mask | (1 << i)]-values[mask | (1 << j)]))))
    write_csv(out / 'all_coalitions.csv', rows)
    distance = np.array([r['embedding_distance'] for r in rows])
    difference = np.array([r['marginal_contribution_diff'] for r in rows])
    nonempty = np.array([r['coalition_mask'] != 0 for r in rows])
    local = (distance >= 0) & (distance <= .4)
    report = dict(embedding_model='text-embedding-3-small', dimensions=256,
        all_coalition_rows=len(rows), legacy_nonempty_coalition_rows=int(nonempty.sum()),
        local_fraction_below_2_5_distance_all=float(np.mean(difference[local] <= 2.5*distance[local])),
        local_fraction_below_2_5_distance_legacy=float(np.mean(difference[local & nonempty] <= 2.5*distance[local & nonempty])),
        plotted_rows=5000, plot_sampling_seed=20260906,
        interpretation='Assumption-check cache only. The saved historical output omits empty coalitions. The original plotted sample seed was not saved; this figure is a fresh deterministic plot.')
    (out / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(out / '.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    choices = np.random.default_rng(20260906).choice(np.flatnonzero(nonempty), 5000, replace=False)
    fig, ax = plt.subplots(figsize=(7, 4.8), layout='constrained')
    ax.scatter(distance[choices], difference[choices], s=5, alpha=.2, color='#225e91', rasterized=True)
    ax.plot([0,.4], [0,1], color='#b3482b', label='2.5 × distance (local range)')
    ax.set(xlabel='Cosine distance of singleton summaries', ylabel='Absolute marginal contribution difference',
        title='Assumption check: saved small-model distances')
    ax.legend(frameon=False)
    fig.savefig(out / 'lipschitz_check.png', dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
