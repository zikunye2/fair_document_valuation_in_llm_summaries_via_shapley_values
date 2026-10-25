#!/usr/bin/env python3
"""Reconstruct Figure A6 from all pair--coalition combinations in the saved cache.

The default scatter uses the PDF-encoded marker placements matched to existing
cache rows, restoring the historical raw distances, style and axis limits. If
the optional archival selection is absent, it draws a fixed-seed legacy sample.
A separate full-range diagnostic samples all coalitions with corrected distances.
Raw cached distances are retained alongside cosine distances clipped to [0, 2],
so floating-point roundoff below zero cannot remove identical-summary pairs.
"""
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
    negative_distance_pairs = 0
    for item in mapping:
        query = int(item['legacy_query_id'])
        values, _ = load_game(PACKAGE / 'data' / 'coalition_games_8doc' / Path(item['original_coalition_csv']).name)
        matrix = np.load(PACKAGE / 'data' / 'appendix' / 'assumption_check' / 'distances' / Path(item['original_distance_matrix']).name, allow_pickle=False)
        if matrix.shape != (8, 8) or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
            raise ValueError(f'Invalid distance matrix for query {query}')
        for i in range(8):
            for j in range(i+1,8):
                raw_distance = float(matrix[i,j])
                distance = float(np.clip(raw_distance, 0, 2))
                negative_distance_pairs += int(raw_distance < 0)
                for mask in range(256):
                    if mask & ((1 << i) | (1 << j)):
                        continue
                    rows.append(dict(query_id=query, doc_i=i, doc_j=j, coalition_mask=mask,
                        coalition_size=mask.bit_count(), raw_embedding_distance=raw_distance,
                        embedding_distance=distance,
                        marginal_contribution_diff=float(abs(values[mask | (1 << i)]-values[mask | (1 << j)]))))
    write_csv(out / 'all_coalitions.csv', rows)
    distance = np.array([r['embedding_distance'] for r in rows])
    raw_distance = np.array([r['raw_embedding_distance'] for r in rows])
    difference = np.array([r['marginal_contribution_diff'] for r in rows])
    nonempty = np.array([r['coalition_mask'] != 0 for r in rows])
    local = (distance >= 0) & (distance <= .4)
    raw_local = (raw_distance >= 0) & (raw_distance <= .4)
    full_range_xlim = [0, float(max(1.05, np.ceil(distance.max() * 20) / 20))]
    historical_xlim = [.01, 1]
    sampling_seed = 20260906
    choices = np.random.default_rng(sampling_seed).choice(np.flatnonzero(nonempty), 5000, replace=False)
    historical_seed = sampling_seed
    historical_source = 'fixed-seed sample of the legacy nonempty-coalition frame'
    archival_selection = {}
    selection_path = PACKAGE / 'data/appendix/assumption_check/figure_a6_plot_selection.csv'
    if selection_path.exists():
        with selection_path.open() as stream:
            selected = sorted(csv.DictReader(stream), key=lambda row: int(row['recovered_plot_order_zero_based']))
        choices = np.array([int(row['matched_all_coalitions_row_index_zero_based']) for row in selected])
        if (len(selected) != 4955 or len(set(choices)) != len(selected)
            or [int(row['recovered_plot_order_zero_based']) for row in selected] != list(range(len(selected)))
            or np.any(choices < 0) or np.any(choices >= len(rows))):
            raise ValueError('Expected 4,955 ordered, distinct archival source-row assignments')
        for selected_row, index in zip(selected, choices):
            if (not nonempty[index]
                or any(int(selected_row[key]) != rows[index][key]
                       for key in ['query_id', 'doc_i', 'doc_j', 'coalition_mask', 'coalition_size'])
                or abs(float(selected_row['embedding_distance']) - raw_distance[index]) > 1e-10
                or abs(float(selected_row['marginal_contribution_diff']) - difference[index]) > 1e-10):
                raise ValueError('Archival plot selection does not match the current saved cache')
        metadata_path = selection_path.with_name('figure_a6_plot_recovery.json')
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        ambiguous = sum(row['row_identity_ambiguous'].lower() == 'true' for row in selected)
        archival_selection = dict(selection_file=str(selection_path.relative_to(PACKAGE)),
            source_pdf_sha256=metadata.get('original_pdf_sha256'), encoded_marker_placements=len(selected),
            paper_stated_sample_size=5000, unencoded_input_points_not_invented=5000-len(selected),
            row_identity_ambiguous_count=ambiguous, row_identity_unique_count=len(selected)-ambiguous,
            max_cache_coordinate_difference=max(max(abs(float(row['embedding_distance'])-raw_distance[index]),
                abs(float(row['marginal_contribution_diff'])-difference[index]))
                for row, index in zip(selected, choices)),
            interpretation='PDF-encoded marker placements recovered; matching cache rows reproduce the geometry, but ambiguous original observation identities are not uniquely recovered.')
        historical_seed = None
        historical_source = 'archived PDF-encoded marker placements matched to cache'
    full_range_choices = np.random.default_rng(sampling_seed).choice(len(rows), 5000, replace=False)

    def save_points(filename, indices, plot_distance, xlim):
        visible = ((plot_distance[indices] >= xlim[0]) & (plot_distance[indices] <= xlim[1])
                   & (difference[indices] >= 0) & (difference[indices] <= 10))
        write_csv(out / filename,
                  [dict(plot_order_zero_based=order,
                        all_coalitions_row_index_zero_based=int(index),
                        plot_x=float(plot_distance[index]), plot_y=float(difference[index]),
                        in_plot_viewport=bool(visible[order]), **rows[index])
                   for order, index in enumerate(indices)])
        return dict(sampled_rows=len(indices), visible_sample_rows=int(visible.sum()),
                    cropped_sample_rows=int((~visible).sum()),
                    sample_rows_below_x_min=int((plot_distance[indices] < xlim[0]).sum()),
                    sample_rows_above_x_max=int((plot_distance[indices] > xlim[1]).sum()))

    historical_view_counts = save_points('plotted_points.csv', choices, raw_distance, historical_xlim)
    full_range_view_counts = save_points('full_range_plotted_points.csv', full_range_choices,
                                         distance, full_range_xlim)
    report = dict(embedding_model='text-embedding-3-small', dimensions=256,
        all_coalition_rows=len(rows), legacy_nonempty_coalition_rows=int(nonempty.sum()),
        distance_policy='embedding_distance = clip(raw_embedding_distance, 0, 2); raw cached values are retained',
        raw_negative_distance_pairs=negative_distance_pairs,
        raw_negative_distance_rows=int((raw_distance < 0).sum()),
        raw_distance_min=float(raw_distance.min()), raw_distance_max=float(raw_distance.max()),
        distance_min=float(distance.min()), distance_max=float(distance.max()),
        local_distance_interval=[0, .4],
        local_rows_all=int(local.sum()),
        local_rows_legacy_nonempty=int((local & nonempty).sum()),
        local_fraction_below_2_5_distance_all=float(np.mean(difference[local] <= 2.5*distance[local])),
        local_fraction_below_2_5_distance_legacy=float(np.mean(difference[local & nonempty] <= 2.5*distance[local & nonempty])),
        raw_cache_local_rows_all=int(raw_local.sum()),
        raw_cache_local_rows_legacy_nonempty=int((raw_local & nonempty).sum()),
        raw_cache_local_fraction_below_2_5_distance_all=float(np.mean(difference[raw_local] <= 2.5*raw_distance[raw_local])),
        raw_cache_local_fraction_below_2_5_distance_legacy=float(np.mean(difference[raw_local & nonempty] <= 2.5*raw_distance[raw_local & nonempty])),
        plotted_rows=len(choices), plot_sampling_seed=historical_seed,
        default_plot_source=historical_source,
        archival_selection=archival_selection,
        plot_sampling_universe='legacy nonempty-coalition combinations; no outcome filtering',
        plot_sampling_universe_rows=int(nonempty.sum()), plot_sampling_replace=False,
        plot_distance_column='raw_embedding_distance',
        plot_axis_limits=dict(x=historical_xlim, y=[0, 10]),
        historical_plot_axis_limits=dict(x=[.01, 1], y=[0, 10]),
        plotted_points_file='plotted_points.csv',
        historical_view=dict(figure_files=['lipschitz_check.png', 'lipschitz_check.pdf'],
            plotted_points_file='plotted_points.csv',
            sampling_universe='84,672 nonempty-coalition combinations',
            sampling_seed=historical_seed, source=historical_source, distance_column='raw_embedding_distance',
            axis_limits=dict(x=historical_xlim, y=[0, 10]),
            sampling_frame_rows_below_x_min=int((nonempty & (raw_distance < historical_xlim[0])).sum()),
            sampling_frame_rows_above_x_max=int((nonempty & (raw_distance > historical_xlim[1])).sum()),
            **historical_view_counts),
        full_range_diagnostic=dict(figure_files=['lipschitz_check_full_range.png', 'lipschitz_check_full_range.pdf'],
            plotted_points_file='full_range_plotted_points.csv',
            sampling_universe='all 86,016 pair--coalition combinations, including the empty coalition',
            sampling_universe_rows=len(rows), sampling_seed=sampling_seed,
            distance_column='embedding_distance', axis_limits=dict(x=full_range_xlim, y=[0, 10]),
            **full_range_view_counts),
        interpretation='Assumption-check cache only. When the archival selection is present, the default plot recovers the 4,955 PDF-encoded marker placements by matching existing cache coordinates; ambiguous original observation identities and the missing 45 of the stated 5,000 input points are not invented. Without it, the default falls back to a fixed-seed legacy sample. The historical viewport [0.01, 1] crops display only; all rows remain in the full data and statistics. The separate full-range diagnostic samples all coalitions with corrected distances. Corrected full-data fractions remain primary; raw-cache fractions are provenance only. Neither view establishes the universal Lipschitz assumption or reproduces the historical claim of more than 95%.')
    (out / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(out / '.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(raw_distance[choices], difference[choices], alpha=.4, s=8, color='blue')
    ax.set(xlim=historical_xlim, ylim=[0, 10])
    ax.set_xlabel('Embedding Distance (1 - cosine similarity)', fontsize=12)
    ax.set_ylabel('|Marginal Contribution Difference|', fontsize=12)
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(out / 'lipschitz_check.png', dpi=180, bbox_inches='tight')
    fig.savefig(out / 'lipschitz_check.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.8), layout='constrained')
    ax.scatter(distance[full_range_choices], difference[full_range_choices], s=5, alpha=.2,
               color='#225e91', rasterized=True)
    ax.set(xlabel='Cosine distance of singleton summaries', ylabel='Absolute marginal contribution difference',
        title='Assumption check: full-range diagnostic', xlim=full_range_xlim, ylim=[0, 10])
    fig.savefig(out / 'lipschitz_check_full_range.png', dpi=180)
    fig.savefig(out / 'lipschitz_check_full_range.pdf', dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
