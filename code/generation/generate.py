#!/usr/bin/env python3
"""Optional paid regeneration. Omitting --allow-api always performs a dry run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from api_client import APIClient, APIError, now
from inputs import coalition_keys, load_input, write_game
import prompts


ORIGINAL_MODEL = "gpt-4o-2024-08-06"
REVISION_SUMMARIZER = "google/gemini-2.5-flash-lite"
REVISION_EVALUATOR = "google/gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-3-large"
PASSES = 4
SUMMARY_SCHEMA = {"type": "object", "properties": {
    "key": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["key", "summary"], "additionalProperties": False}
EVALUATION_SCHEMA = {"type": "object", "properties": {"evaluations": {
    "type": "array", "items": {"type": "object", "properties": {
        "key": {"type": "string"}, "score": {"type": "integer", "enum": list(range(11))}},
        "required": ["key", "score"], "additionalProperties": False}}},
    "required": ["evaluations"], "additionalProperties": False}


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def parse_relevance(text, count):
    labels = {}
    for index, relevant in re.findall(r"\[(\d+)\] is (relevant to|not related to) the query\.", text):
        index = int(index)
        if index in labels:
            raise ValueError("Duplicate relevance label; model response retained for inspection")
        labels[index] = relevant == "relevant to"
    if set(labels) != set(range(count)):
        raise ValueError("Incomplete relevance labels; refusing to infer missing labels as irrelevant")
    return [labels[i] for i in range(count)]


def parse_scores(text, expected):
    result = parse_json(text)
    if isinstance(result, dict):
        candidates = [value for value in result.values() if isinstance(value, list)]
        if len(candidates) != 1:
            raise ValueError("Evaluation must contain exactly one score array")
        result = candidates[0]
    if not isinstance(result, list):
        raise ValueError("Evaluation must return a score array")
    scores = {}
    for row in result:
        if not isinstance(row, dict) or not isinstance(row.get("key"), str):
            raise ValueError("Malformed evaluation row")
        key, score = row["key"], row.get("score")
        if key in scores or type(score) is not int or not 0 <= score <= 10:
            raise ValueError("Duplicate or invalid evaluation score")
        scores[key] = score
    if set(scores) != set(expected):
        raise ValueError("Missing or unexpected coalition scores; no zero fallback is permitted")
    return scores


def label_documents(client, documents, query, call_id, *, revision):
    prompt = prompts.revision_relevance if revision else prompts.original_relevance
    model = REVISION_SUMMARIZER if revision else ORIGINAL_MODEL
    user = "\n\n".join(f"[{i}] {document}" for i, document in enumerate(documents))
    return parse_relevance(client.chat(call_id, model, prompt(query), user), len(documents))


def summarize(client, documents, labels, query, call_id, *, revision):
    relevant = [document for document, label in zip(documents, labels) if label]
    irrelevant = [str(i) for i, label in enumerate(labels) if not label]
    local_key = "".join(str(i) for i in range(len(documents)))
    if not relevant:
        return {"key": local_key,
                "summary": " ".join(f"[{i}] is not related to the query." for i in irrelevant)}
    prompt = prompts.revision_summary if revision else prompts.original_summary
    model = REVISION_SUMMARIZER if revision else ORIGINAL_MODEL
    text = client.chat(call_id, model, prompt(query), "\n\n".join(relevant),
                       schema=SUMMARY_SCHEMA, json_mode=revision)
    output = parse_json(text)
    if not isinstance(output, dict) or not isinstance(output.get("summary"), str) or not output["summary"].strip():
        raise ValueError("Missing summary text; raw response is retained")
    # Preserve historical post-processing and subset-local citation indices.
    summary = output["summary"]
    if irrelevant:
        summary += " " + " ".join(f"[{i}] is not related to the query." for i in irrelevant)
    return {"key": local_key, "summary": summary}


def evaluate_four(client, summaries, query, arm, *, revision):
    keys = list(summaries)
    chunk = 300 if revision else len(keys)
    prompt = prompts.revision_evaluation if revision else prompts.original_evaluation
    model = REVISION_EVALUATOR if revision else ORIGINAL_MODEL
    passes = []
    for iteration in range(PASSES):
        scores = {}
        for start in range(0, len(keys), chunk):
            selected = keys[start:start+chunk]
            user = "\n\n".join(f"Summary[{key}]: {summaries[key]['summary']}" for key in selected)
            text = client.chat(f"{arm}:evaluation:{iteration+1}:chunk:{start//chunk+1}",
                               model, prompt(query), user,
                               schema=EVALUATION_SCHEMA, json_mode=revision)
            scores.update(parse_scores(text, selected))
        passes.append(scores)
    averages = {key: sum(scores[key] for scores in passes)/PASSES for key in keys}
    return averages, passes


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_original(client, data, output_dir):
    summaries, labels_by_subset = {}, {}
    for key in coalition_keys(len(data["documents"])):
        documents = [data["documents"][int(i)] for i in key]
        labels = label_documents(client, documents, data["query"], f"original:relevance:{key}", revision=False)
        labels_by_subset[key] = labels
        summaries[key] = summarize(client, documents, labels, data["query"],
                                   f"original:summary:{key}", revision=False)
    scores, per_pass = evaluate_four(client, summaries, data["query"], "original", revision=False)
    phi = write_game(output_dir/data["dataset_name"], data, summaries, scores)
    save_json(output_dir/"generated_game.json", {
        "interpretation": "New stochastic game, not recovery of the historical exact game.",
        "input": data, "summaries": summaries, "per_subset_relevance": labels_by_subset,
        "scores_by_pass": per_pass, "average_scores": scores, "empty_coalition_value": 0,
        "exact_shapley": phi, "completed_at_utc": now()})


def run_robustness(client, data, output_dir):
    # The archived CSV row order is preserved for comparative evaluation, as in exp22.
    cached = {key: {"summary": summary} for key, summary in data["summaries"].items()}
    first_scores, first_passes = evaluate_four(client, cached, data["query"], "cached-gpt4o", revision=True)
    labels = label_documents(client, data["documents"], data["query"], "gemini:full-set-relevance", revision=True)
    regenerated = {}
    for key in coalition_keys(len(data["documents"])):
        documents = [data["documents"][int(i)] for i in key]
        subset_labels = [labels[int(i)] for i in key]
        regenerated[key] = summarize(client, documents, subset_labels, data["query"],
                                     f"gemini:summary:{key}", revision=True)
    second_scores, second_passes = evaluate_four(client, regenerated, data["query"], "gemini", revision=True)
    first_phi = write_game(output_dir/"cached_gpt4o_rescored.csv", data, cached, first_scores)
    second_phi = write_game(output_dir/"gemini_regenerated.csv", data, regenerated, second_scores)
    save_json(output_dir/"summarizer_robustness.json", {
        "interpretation": "New stochastic robustness run; original saved scores are not overwritten.",
        "input": data, "full_set_relevance": labels, "generated_summaries": regenerated,
        "cached_gpt4o_scores_by_pass": first_passes, "gemini_scores_by_pass": second_passes,
        "v1": first_scores, "v2": second_scores, "empty_coalition_value": 0,
        "cached_gpt4o_exact_shapley": first_phi, "gemini_exact_shapley": second_phi,
        "completed_at_utc": now()})


def run_embeddings(client, datasets, output_dir):
    import numpy as np
    manifest = {"model": EMBEDDING_MODEL, "dimensions": 3072, "input_text": "singleton summaries",
                "source_note": "New API collection, not the missing historical embedding cache.",
                "collected_at_utc": now(), "files": {}, "records": {}}
    for data in datasets:
        texts = [data["summaries"][str(i)] for i in range(len(data["documents"]))]
        raw = client.call("embeddings:"+data["dataset_name"], "/embeddings", {
            "model": EMBEDDING_MODEL, "dimensions": 3072, "encoding_format": "float", "input": texts})
        rows = raw["data"]
        if len(rows) != len(texts) or {r["index"] for r in rows} != set(range(len(texts))):
            raise ValueError("Embedding response has missing or duplicate document indices")
        vectors = np.array([r["embedding"] for r in sorted(rows, key=lambda r: r["index"])], dtype=np.float64)
        if vectors.shape != (len(texts), 3072) or not np.isfinite(vectors).all():
            raise ValueError("Embedding vectors must be finite with exactly 3072 dimensions")
        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("A zero embedding cannot define cosine distance")
        filename = data["dataset_name"] + ".embeddings.npy"
        np.save(output_dir/filename, vectors, allow_pickle=False)
        manifest["files"][data["dataset_name"]] = filename
        manifest["records"][data["dataset_name"]] = {
            "query": data["query"], "input_sha256": data["input_sha256"],
            "metadata_warnings": data["metadata_warnings"],
            "document_indices": list(range(len(texts))),
            "singleton_summary_sha256": [hashlib.sha256(t.encode()).hexdigest() for t in texts],
            "returned_model": raw.get("model"), "response_id": raw.get("id"),
            "array_sha256": hashlib.sha256((output_dir/filename).read_bytes()).hexdigest()}
    save_json(output_dir/"embedding_manifest.json", manifest)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("original", "robustness", "embeddings"):
        command = commands.add_parser(name)
        command.add_argument("--provider", required=True,
                             choices=["openrouter"] if name == "robustness" else ["openai"])
        command.add_argument("--input", required=True, type=Path,
                             help="One CSV/JSON; embeddings also accepts a directory of CSVs")
        command.add_argument("--output-dir", required=True, type=Path)
        command.add_argument("--allow-api", action="store_true", help="Explicitly permit paid calls")
        command.add_argument("--resume", action="store_true", help="Reuse matching response journal")
        if name == "embeddings":
            command.add_argument("--max-products", type=int, default=1,
                                 help="Maximum CSV files from a directory; default 1")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "embeddings" and args.max_products < 1:
        parser.error("--max-products must be positive")
    if args.input.is_dir():
        if args.command != "embeddings":
            parser.error("original and robustness accept one input game at a time")
        paths = sorted(args.input.glob("*.csv"))[:args.max_products]
    else:
        paths = [args.input]
    if not paths:
        parser.error("No input CSV files found")
    datasets = [load_input(path, require_summaries=args.command != "original") for path in paths]
    for data in datasets:
        for warning in data["metadata_warnings"]:
            print(f"Input metadata warning for {data['dataset_name']}: {warning}")
    if len({data["dataset_name"] for data in datasets}) != len(datasets):
        parser.error("Input dataset filenames must be unique")
    counts = [len(coalition_keys(len(data["documents"]))) for data in datasets]
    calls = (sum(counts)*2+PASSES if args.command == "original" else
             sum(1+count+2*PASSES*((count+299)//300) for count in counts)
             if args.command == "robustness" else len(datasets))
    config = {"schema_version": 1, "command": args.command, "provider": args.provider,
              "datasets": [{"filename": data["dataset_name"], "query": data["query"],
                            "metadata_warnings": data["metadata_warnings"],
                            "input_sha256": data["input_sha256"]} for data in datasets],
              "original_model": ORIGINAL_MODEL, "revision_summarizer": REVISION_SUMMARIZER,
              "revision_evaluator": REVISION_EVALUATOR, "embedding_model": EMBEDDING_MODEL,
              "temperature": 0.1, "evaluation_passes": PASSES,
              "revision_max_tokens": 32768, "revision_evaluation_chunk": 300,
              "prompt_source_sha256": hashlib.sha256(Path(prompts.__file__).read_bytes()).hexdigest(),
              "new_stochastic_collection": True}
    print(f"{args.command}: {len(datasets)} game(s), at most {calls} API call(s); no automatic paid retries.")
    if not args.allow_api:
        print("Dry run only. No network request, credential read, or output write. Add --allow-api to execute.")
        return 0
    output = args.output_dir.resolve()
    config_path = output/"run_config.json"
    if args.resume:
        if not config_path.exists() or json.loads(config_path.read_text()) != config:
            parser.error("Resume requires an existing output directory with exactly matching run configuration")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_json(config_path, config)
    client = APIClient(args.provider, output, allow_api=True)
    if args.command == "original":
        run_original(client, datasets[0], output)
    elif args.command == "robustness":
        run_robustness(client, datasets[0], output)
    else:
        run_embeddings(client, datasets, output)
    print(f"Completed new collection in {output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (APIError, ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
        # Validation messages are authored here; HTTP messages are sanitized in APIClient.
        if isinstance(error, (KeyError, json.JSONDecodeError)):
            message = "Malformed input or model output. Inspect the input and response journal."
        else:
            message = str(error)
        print(f"Stopped: {message}", file=sys.stderr)
        sys.exit(1)
