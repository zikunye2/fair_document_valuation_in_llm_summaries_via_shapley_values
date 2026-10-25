#!/usr/bin/env python3
"""Rebuild docs/DATA_MANIFEST.csv (path, source, role, bytes, sha256 of every file in data/).

Maintainer tool. The source and role columns of files already listed are kept; new files get
the source and role given by --source and --role (or the defaults below for known folders).
run_all.py's first stage (code/verify_inventory.py) checks data/ against this manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs' / 'DATA_MANIFEST.csv'
DEFAULT_ROLES = {
    'data/archived_runs/': ('organized.zip from Yizhuo Chang, 2026-09-14',
                            'Archived notebook run results behind Figure 3 and Table 2 Panel B'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='', help='source text for files not yet in the manifest')
    parser.add_argument('--role', default='', help='role text for files not yet in the manifest')
    args = parser.parse_args()
    known = {}
    if MANIFEST.exists():
        with MANIFEST.open(newline='') as stream:
            for row in csv.DictReader(stream):
                known[row['file']] = (row['source'], row['role'])
    rows = []
    for path in sorted(p for p in (ROOT / 'data').rglob('*') if p.is_file() and p.name != '.DS_Store'):
        relative = path.relative_to(ROOT).as_posix()
        payload = path.read_bytes()
        source, role = known.get(relative, (args.source, args.role))
        if not source:
            for prefix, (s, r) in DEFAULT_ROLES.items():
                if relative.startswith(prefix):
                    source, role = s, r
        rows.append(dict(file=relative, source=source, role=role, bytes=len(payload),
                         sha256=hashlib.sha256(payload).hexdigest()))
    with MANIFEST.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['file', 'source', 'role', 'bytes', 'sha256'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {MANIFEST.relative_to(ROOT)} with {len(rows)} files, '
          f'{sum(r["bytes"] for r in rows)} bytes')


if __name__ == '__main__':
    main()
