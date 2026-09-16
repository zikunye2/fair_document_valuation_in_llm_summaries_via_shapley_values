#!/usr/bin/env python3
"""Main-text table `tab:filtered_reviews_shapley_value`: exact Shapley values of the
eight retrieved reviews in the wireless-controller example.

Inputs, all offline:
  data/coalition_games_8doc/Shapley_the quality of the product_switch.csv
      The archived coalition game of the example (the review text identifies the
      wireless controller; the legacy filename says switch): review texts, averaged
      coalition scores and the stored document values.
Output (outputs/main_text/):
  table1_example_shapley.csv   per review: text, stored value, exact value recomputed from
                              the archived scores

Use --dataset to select another archived coalition game.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / "code" / "algorithms"))
sys.path.insert(0, str(PACKAGE / "code" / "generation"))
from run_baselines import exact_shapley, load_game, write_csv  # noqa: E402
from inputs import load_input  # noqa: E402

DEFAULT_DATASET = "Shapley_the quality of the product_switch.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE / "data/coalition_games_8doc")
    parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATASET),
                        help="Coalition CSV filename relative to --data-dir, or an absolute path")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "outputs/main_text")
    args = parser.parse_args()
    path = args.data_dir / args.dataset
    values, stored = load_game(path)
    exact = exact_shapley(values)
    documents = load_input(path)["documents"]
    if len(documents) != len(stored):
        raise SystemExit("Document count mismatch within the archived game")
    rows = []
    for i, text in enumerate(documents):
        rows.append({"document_one_indexed": i + 1, "review_text": text,
                     "stored_value": float(stored[i]), "recomputed_exact_value": float(exact[i])})
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "table1_example_shapley.csv", rows)
    print(f"{path.name}: full-set score {values[-1]:g}")
    print(f"{'doc':>4}{'stored':>10}{'exact':>10}  review (first 60 characters)")
    for r in rows:
        print(f"{r['document_one_indexed']:>4}{r['stored_value']:10.3f}{r['recomputed_exact_value']:10.3f}"
              f"  {r['review_text'][:60]!r}")
    print(f"Wrote {out / 'table1_example_shapley.csv'}")


if __name__ == "__main__":
    main()
