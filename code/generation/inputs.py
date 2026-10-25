"""Validate retrieved-document inputs and write complete cooperative games."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path


def coalition_keys(n):
    return ["".join(map(str, subset)) for size in range(1, n + 1)
            for subset in combinations(range(n), size)]


def load_input(path, *, require_summaries=False):
    path = Path(path)
    warnings = []
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        query = data["query"]
        documents = data["documents"]
        name = data["dataset_name"]
        product = data.get("product_description", "")
        cached = data.get("summaries", {})
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("Input CSV is empty")
        query = rows[0]["original_query"]
        distinct_queries = {row["original_query"] for row in rows}
        if len(distinct_queries) != 1:
            warnings.append(f"CSV has {len(distinct_queries)} distinct row-level original_query values; "
                            "the first-row query is used, matching the historical revision loader. "
                            "Archived row metadata is not rewritten.")
        product = rows[0].get("product_description", "")
        name = path.name
        singles, cached = {}, {}
        for row in rows:
            key = row["subset"]
            if key.startswith("Comment"):
                continue
            if key in cached:
                raise ValueError("Duplicate coalition in CSV")
            cached[key] = row["summary"]
            if len(key) == 1 and key.isascii() and key.isdigit():
                comments = ast.literal_eval(row["comments"])
                if not isinstance(comments, list) or len(comments) != 1:
                    raise ValueError("Singleton comments must be a one-element list")
                singles[int(key)] = comments[0]
        if set(singles) != set(range(len(singles))):
            raise ValueError("Document indices must be contiguous from zero")
        documents = [singles[i] for i in range(len(singles))]
    else:
        raise ValueError("Input must be a retrieved-document JSON or archived game CSV")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be nonempty text")
    if not isinstance(documents, list) or not 1 <= len(documents) <= 10:
        raise ValueError("Supply 1 to 10 retrieved documents; retrieval is outside this script")
    if any(not isinstance(document, str) or not document.strip() for document in documents):
        raise ValueError("Every retrieved document must be nonempty text")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".csv"):
        raise ValueError("dataset_name must be a plain CSV filename")
    if not isinstance(cached, dict) or any(not isinstance(s, str) for s in cached.values()):
        raise ValueError("Cached summaries must map coalition keys to text")
    if require_summaries and set(cached) != set(coalition_keys(len(documents))):
        raise ValueError("This operation requires all nonempty cached coalition summaries")
    return {"dataset_name": name, "query": query, "documents": documents,
            "product_description": product, "summaries": cached,
            "metadata_warnings": warnings,
            "input_filename": path.name,
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def exact_shapley(scores, n):
    expected = set(coalition_keys(n))
    if set(scores) != expected:
        raise ValueError("Exact Shapley requires every nonempty coalition score")
    values = {0: 0.0}
    values.update({sum(1 << int(i) for i in key): float(value) for key, value in scores.items()})
    output = []
    for i in range(n):
        total = 0.0
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(n-size-1) / math.factorial(n)
            total += weight * (values[mask | (1 << i)] - values[mask])
        output.append(total)
    return output


def write_game(path, data, summaries, scores):
    documents = data["documents"]
    phi = exact_shapley(scores, len(documents))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["original_query", "product_description",
                                                   "subset", "comments", "summary", "avg_score"])
        writer.writeheader()
        common = {"original_query": data["query"], "product_description": data["product_description"]}
        for key in coalition_keys(len(documents)):
            writer.writerow({**common, "subset": key,
                             "comments": repr([documents[int(i)] for i in key]),
                             "summary": summaries[key]["summary"], "avg_score": scores[key]})
        for i, value in enumerate(phi):
            writer.writerow({**common, "subset": f"Comment {i}", "comments": documents[i],
                             "summary": "", "avg_score": value})
    return phi
