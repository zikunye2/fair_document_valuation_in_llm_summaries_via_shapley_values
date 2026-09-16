#!/usr/bin/env python3
"""Convert authenticated/source-verified singleton-summary embedding arrays to distances.

This is an offline converter. Embeddings must be supplied separately; no API
requests or credential access are performed. Row i must embed singleton i's
summary from the corresponding coalition CSV, not the original review text.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from run_baselines import PACKAGE, load_game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-manifest", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE / "data/coalition_games_8doc")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "outputs/algorithms/distances")
    args = parser.parse_args()
    source = args.embedding_manifest.resolve()
    manifest = json.loads(source.read_text())
    if manifest.get("input_text") != "singleton summaries":
        raise ValueError("Manifest input_text must be singleton summaries")
    if not isinstance(manifest.get("dimensions"), int) or not manifest.get("model"):
        raise ValueError("Manifest requires model and integer dimensions")
    paths = sorted(args.data_dir.resolve().glob("*.csv"))
    if not paths or set(manifest["files"]) != {p.name for p in paths}:
        raise ValueError("Manifest must map all and only the supplied CSVs")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for idx, path in enumerate(paths):
        _, stored = load_game(path)
        embeddings = np.load(source.parent / manifest["files"][path.name], allow_pickle=False)
        if embeddings.shape != (len(stored), manifest["dimensions"]) or not np.all(np.isfinite(embeddings)):
            raise ValueError(f"{path.name}: embedding dimensions or values invalid")
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms == 0):
            raise ValueError(f"{path.name}: zero embedding cannot define cosine distance")
        unit = embeddings / norms[:, None]
        distances = np.clip(1 - unit @ unit.T, 0, 2)
        np.fill_diagonal(distances, 0)
        filename = f"distance_{idx:03d}.npy"
        np.save(out / filename, distances, allow_pickle=False)
        mapping[path.name] = filename
    result = {"model": manifest["model"], "dimensions": manifest["dimensions"],
              "input_text": "singleton summaries", "files": mapping,
              "source_note": manifest.get("source_note", "User-supplied embeddings; historical identity unverified")}
    (out / "distance_manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"Prepared {len(mapping)} distance matrices: {out / 'distance_manifest.json'}")


if __name__ == "__main__":
    main()
