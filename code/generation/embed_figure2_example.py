#!/usr/bin/env python3
"""Opt-in collection of the two-dimensional embeddings behind Figure 2.

The published Figure 2 places the eight reviews of the wireless-controller example with
two-dimensional embeddings requested directly from the OpenAI API (text-embedding-3-large
with dimensions=2) of the text "review title. single-review summary", and clusters them with
DBSCAN (min_samples=1, eps=0.05) on the cosine distances of those two-dimensional vectors.
This script collects those eight vectors once and stores them in
data/embeddings/figure2_example_2d/embeddings_2d.json, which the Figure 2 scripts read
offline. The eight input texts are fixed here exactly as used for the published figure.

Requires --allow-api and OPENAI_API_KEY in the environment; never run by run_all.py.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_client import APIClient, now  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2]
MODEL = "text-embedding-3-large"
DIMENSIONS = 2
DATASET = "Shapley_the quality of the product_switch.csv"
TEXTS = [
    "Cheaper price, same great quality. [0] is not related to the query.",
    "Quality. The product is considered to be worth its price due to its durability, as the controllers last significantly longer than off-brand alternatives [0].",
    "Great Quality. The product is praised for its great quality and price, with the user noting that it worked well compared to a previous purchase from another seller that was unsatisfactory. The user expresses satisfaction with the seller, rating the product five stars for its legitimacy and performance [0].",
    "Great buy and Product is exactly what I expected. The user expressed satisfaction with the product, highlighting that the quality met their expectations and that they appreciated the red color of the item [0].",
    "Five stars. The user highly recommends purchasing the original maker's product, suggesting that it is worth paying more for better quality compared to knockoffs [0].",
    "great product. [0] is not related to the query.",
    "Nice, new and crispy. The product is described as 'nice, new, and crispy,' with the user expressing high satisfaction with its quality. The vendor and price also receive positive remarks, leading to a strong recommendation with a rating of 10/10 [0].",
    "Quality.[0] is not related to the query.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-api", action="store_true", help="permit the network call")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "data/embeddings/figure2_example_2d")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    client = APIClient("openai", out, allow_api=args.allow_api)
    raw = client.call("figure2_example_2d", "/embeddings",
                      {"model": MODEL, "input": TEXTS, "dimensions": DIMENSIONS, "encoding_format": "float"})
    rows = sorted(raw["data"], key=lambda r: r["index"])
    if len(rows) != len(TEXTS):
        raise SystemExit("Provider returned a different number of embeddings than texts")
    vectors = np.array([r["embedding"] for r in rows], dtype=float)
    if vectors.shape != (len(TEXTS), DIMENSIONS) or not np.all(np.isfinite(vectors)):
        raise SystemExit("Unexpected embedding shape or nonfinite values")
    record = {"dataset": DATASET, "model": raw.get("model", MODEL), "dimensions": DIMENSIONS,
              "collected_at_utc": now(), "collection_date": str(date.today()),
              "input_text": "review title, a period, the single-review summary of the archived game",
              "texts": TEXTS, "embeddings": vectors.tolist(),
              "note": "Two-dimensional embeddings as used for the published Figure 2; the Figure 2 "
                      "scripts cluster them with DBSCAN (min_samples=1, eps=0.05, cosine distance) "
                      "and min-max scale them to [-5, 5] for display."}
    (out / "embeddings_2d.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    distances = np.clip(1 - unit @ unit.T, 0, 2)
    print("two-dimensional embeddings:")
    for i, (x, y) in enumerate(vectors):
        print(f"  review {i + 1}: ({x:+.6f}, {y:+.6f})")
    print("cosine distances below 0.05:",
          [(i + 1, j + 1, round(float(distances[i, j]), 4)) for i in range(len(TEXTS))
           for j in range(i + 1, len(TEXTS)) if distances[i, j] < 0.05])
    print(f"Wrote {out / 'embeddings_2d.json'}")


if __name__ == "__main__":
    main()
