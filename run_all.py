#!/usr/bin/env python3
"""Run the offline replication workflow. No model API calls.

Stages, in order (all read only the files in data/ and write to outputs/):
  1. check_data           checksums of every data file against docs/DATA_MANIFEST.csv
  2. test_algorithms      unit tests of the Shapley algorithms
  3. main text            cosine distances -> Cluster Shapley (41 epsilon values) ->
                          Monte Carlo / Truncated Monte Carlo / Kernel SHAP (48 games, 10 seeds)
                          -> Table 1, Figure 2 (archived game),
                          Figure 3, Table 2 (outputs/main_text/)
  4. archived runs        Figure 3 and Table 2 Panel B redrawn from the archived notebook
                          result files in data/archived_runs/ (outputs/main_text/)
  5. appendix             coalition audits, Tables A3 and A4, Figure A6 (outputs/appendix/);
                          skipped with --skip-appendix

The benchmark stage takes about one minute; everything else takes seconds. Every random
draw uses a fixed seed and the numerical libraries run single-threaded, so repeated runs
in the same environment give identical outputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-appendix', action='store_true',
                        help='Run only the data check, the tests and the main-text stages')
    parser.add_argument('--quick-baselines', action='store_true',
                        help='Benchmark only two games, two seeds and three budgets; the regenerated '
                             'Figure 3 is then a smoke test, not the paper figure')
    args = parser.parse_args()
    logs = ROOT / 'outputs' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # The workflow is offline and receives no API credentials.
    for name in list(env):
        if any(word in name.upper() for word in ('API_KEY', 'TOKEN', 'PASSWORD', 'SECRET')):
            env.pop(name)
    env['MPLCONFIGDIR'] = str(ROOT / 'outputs' / '.matplotlib')
    env['XDG_CACHE_HOME'] = str(ROOT / 'outputs' / '.cache')
    env['PYTHONUNBUFFERED'] = '1'
    # Reproducibility: fixed hash seed and single-threaded numerical libraries.
    env['PYTHONHASHSEED'] = '0'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'

    benchmark = ['code/algorithms/run_baselines.py', '--methods', 'mc', 'tmc', 'kernel',
                 '--output-dir', 'outputs/algorithms/baselines']
    if args.quick_baselines:
        benchmark.append('--quick')
    tasks = [
        ('check_data', ['code/verify_inventory.py']),
        ('test_algorithms', ['code/algorithms/test_algorithms.py']),
        # main text
        ('distances', ['code/algorithms/prepare_distances.py', '--embedding-manifest',
                       'data/embeddings/main_8doc_summaries/embedding_manifest.json',
                       '--output-dir', 'outputs/algorithms/distances']),
        ('cluster_shapley', ['code/algorithms/run_cluster.py', '--distance-manifest',
                             'outputs/algorithms/distances/distance_manifest.json',
                             '--output-dir', 'outputs/algorithms/cluster']),
        ('baselines', benchmark),
        ('table1', ['code/main_text/table1_example_shapley.py']),
        ('figure2', ['code/main_text/figure2_clustering_visualization.py']),
        ('figure3', ['code/main_text/figure3_benchmark_comparison.py']),
        ('table2', ['code/main_text/table2_accuracy_efficiency.py']),
        # archived notebook runs
        ('figure3_archived_runs', ['code/main_text/figure3_from_archived_runs.py']),
        ('table2_panel_b_archived_runs', ['code/main_text/table2_panel_b_from_archived_runs.py']),
    ]
    if not args.skip_appendix:
        tasks += [
            ('appendix_audit_8doc', ['code/algorithms/run_baselines.py', '--audit-only',
                                     '--output-dir', 'outputs/appendix/audit_8doc']),
            ('appendix_audit_10doc', ['code/algorithms/run_baselines.py', '--audit-only',
                                      '--data-dir', 'data/appendix/coalition_games_10doc',
                                      '--output-dir', 'outputs/appendix/audit_10doc']),
            ('appendix_table_a3_a4', ['code/appendix/table_a3_a4_retrieval_depth_summarizer.py']),
            ('appendix_figure_a6', ['code/appendix/figure_a6_lipschitz_check.py']),
        ]
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(), python=sys.version.split()[0],
                  options=vars(args), tasks=[])
    for name, command in tasks:
        print(f'Running {name}...', flush=True)
        start = time.monotonic()
        with (logs / f'{name}.log').open('w') as stream:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        report['tasks'].append(dict(name=name, command=['python', *command],
                                    exit_code=result.returncode,
                                    seconds=round(time.monotonic() - start, 3)))
        if result.returncode:
            (logs / 'run_report.json').write_text(json.dumps(report, indent=2) + '\n')
            raise SystemExit(f'{name} failed; see outputs/logs/{name}.log')
        print(f'Completed {name}.', flush=True)
    report['versions'] = {}
    for name in ('numpy', 'pandas', 'scipy', 'matplotlib', 'scikit-learn', 'shap'):
        try:
            report['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['versions'][name] = None
    (logs / 'run_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Workflow passed. Main-text results: outputs/main_text/; appendix: outputs/appendix/; '
          'logs: outputs/logs/. See docs/RESULTS_MAP.md.')


if __name__ == '__main__':
    main()
