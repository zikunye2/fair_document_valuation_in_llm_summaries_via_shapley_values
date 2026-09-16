#!/usr/bin/env python3
"""Check that the selected data match the byte-for-byte source manifest."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / 'docs' / 'DATA_MANIFEST.csv').open() as stream:
        manifest = list(csv.DictReader(stream))
    seen = set()
    for item in manifest:
        relative = Path(item['file'])
        if relative.is_absolute() or '..' in relative.parts or relative in seen:
            raise ValueError('Unsafe or duplicate manifest path')
        seen.add(relative)
        payload = (ROOT / relative).read_bytes()
        if len(payload) != int(item['bytes']) or hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError(f'Data file differs from manifest: {relative}')
    actual = {p.relative_to(ROOT) for p in (ROOT / 'data').rglob('*') if p.is_file() and p.name != '.DS_Store'}
    if actual != seen:
        raise ValueError('Data directory and manifest membership differ')
    print(json.dumps(dict(verified_files=len(seen),bytes=sum(int(r['bytes']) for r in manifest)),indent=2))


if __name__ == '__main__':
    main()
