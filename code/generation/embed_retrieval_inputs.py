#!/usr/bin/env python3
"""Opt-in collection of text-embedding-3-large vectors for retrieved review texts and queries.

The relevance-weighted attribution rule in the main text uses the retrieval relevance
score s_i(q) = cosine_similarity(e_i, e_q) between each retrieved review and the query.
The original retrieval embeddings were not archived. This script embeds, for every
archived game CSV, the eight retrieved review strings exactly as stored in the CSV
("title: ...\ntext: ..." format) and the archived query string, with the same OpenAI
model the manuscript names. OpenAI embeddings are close to deterministic, but the
original retrieval step may have embedded a differently formatted review string or a
differently processed query, so the resulting similarities are an approximation of the
historical retrieval scores, not a recovery of them.

Without --allow-api the script validates inputs and exits. It never prints or stores
the credential; the OpenAI key is read from the OPENAI_API_KEY environment variable of
this process only at call time (see api_client.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_client import APIClient, APIError, now  # noqa: E402
from inputs import load_input  # noqa: E402

EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSIONS = 3072


def embed(client, call_id, texts):
    raw = client.call(call_id, "/embeddings", {
        "model": EMBEDDING_MODEL, "dimensions": DIMENSIONS,
        "encoding_format": "float", "input": texts})
    rows = raw["data"]
    if len(rows) != len(texts) or {r["index"] for r in rows} != set(range(len(texts))):
        raise ValueError("Embedding response has missing or duplicate indices")
    vectors = np.array([r["embedding"] for r in sorted(rows, key=lambda r: r["index"])],
                       dtype=np.float64)
    if vectors.shape != (len(texts), DIMENSIONS) or not np.isfinite(vectors).all():
        raise ValueError("Embedding vectors must be finite with exactly 3072 dimensions")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("A zero embedding cannot define cosine similarity")
    return vectors, raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path,
                        help="Directory of archived coalition-game CSVs")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-api", action="store_true", help="Explicitly permit paid calls")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse an existing output directory and its response journal")
    args = parser.parse_args(argv)
    paths = sorted(args.input.glob("*.csv"))
    if not paths:
        parser.error("No input CSV files found")
    datasets = [load_input(path) for path in paths]
    for data in datasets:
        for warning in data["metadata_warnings"]:
            print(f"Input metadata warning for {data['dataset_name']}: {warning}")
    print(f"retrieval inputs: {len(datasets)} game(s), at most {2 * len(datasets)} API call(s); "
          "no automatic paid retries.")
    if not args.allow_api:
        print("Dry run only. No network request, credential read, or output write. "
              "Add --allow-api to execute.")
        return 0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    client = APIClient("openai", output, allow_api=True)
    manifest = {"model": EMBEDDING_MODEL, "dimensions": DIMENSIONS,
                "input_text": "retrieved review strings as archived in the coalition CSV "
                              "(title and text) and the archived first-row query string",
                "source_note": "New API collection (2026); the original retrieval embeddings "
                               "were not archived. Similarities are an approximation of the "
                               "historical retrieval scores.",
                "collected_at_utc": now(), "files": {}, "records": {}}
    for data in datasets:
        name = data["dataset_name"]
        reviews, raw_reviews = embed(client, "reviews:" + name, data["documents"])
        query, raw_query = embed(client, "query:" + name, [data["query"]])
        review_file = name + ".reviews.npy"
        query_file = name + ".query.npy"
        np.save(output / review_file, reviews, allow_pickle=False)
        np.save(output / query_file, query[0], allow_pickle=False)
        manifest["files"][name] = {"reviews": review_file, "query": query_file}
        manifest["records"][name] = {
            "query": data["query"], "input_sha256": data["input_sha256"],
            "metadata_warnings": data["metadata_warnings"],
            "review_sha256": [hashlib.sha256(t.encode()).hexdigest() for t in data["documents"]],
            "query_sha256": hashlib.sha256(data["query"].encode()).hexdigest(),
            "returned_model": raw_reviews.get("model"),
            "response_ids": [raw_reviews.get("id"), raw_query.get("id")]}
    (output / "retrieval_embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Completed new collection in {output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (APIError, ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        sys.exit(1)
