"""Offline tests. Network access is always patched out or replaced by fake output."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from api_client import APIClient, APIError
import generate
from inputs import coalition_keys, load_input, exact_shapley


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, call_id, model, system, user, **options):
        self.calls.append((call_id, model, user, options))
        if "relevance" in call_id:
            count = sum(line.startswith("[") for line in user.splitlines())
            return " ".join(f"[{i}] is relevant to the query." for i in range(count))
        if "evaluation" in call_id:
            keys = [line.split("]:", 1)[0][8:] for line in user.splitlines() if line.startswith("Summary[")]
            return json.dumps({"evaluations": [{"key": key, "score": len(key)} for key in keys]})
        return json.dumps({"key": "0", "summary": "A mock supported statement [0]."})

    def call(self, call_id, path, body):
        self.calls.append((call_id, path, body))
        return {"model": generate.EMBEDDING_MODEL, "id": "offline-fixture", "data": [
            {"index": i, "embedding": [float(i+1)] + [0.0]*3071}
            for i in range(len(body["input"]))]}


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root/"input.json"
        self.input.write_text(json.dumps({"dataset_name": "Shapley_fixture.csv", "query": "durability",
                                          "documents": ["First review", "Second review"],
                                          "summaries": {"0": "One", "1": "Two", "01": "Both"}}))

    def test_dry_run_never_reads_credential_or_writes(self):
        def reject_credential(name, default=None):
            if name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
                raise AssertionError("credential read forbidden")
            return default
        with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")), \
             patch("api_client.os.environ.get", side_effect=reject_credential):
            self.assertEqual(generate.main(["original", "--provider", "openai", "--input", str(self.input),
                                            "--output-dir", str(self.root/"dry")]), 0)
        self.assertFalse((self.root/"dry").exists())

    def test_client_requires_opt_in_before_reading_credentials(self):
        client = APIClient("openai", self.root)
        with patch("api_client.os.environ.get", side_effect=AssertionError("credential read forbidden")):
            with self.assertRaises(APIError):
                client._post("/chat/completions", {})

    def test_incomplete_and_duplicate_scores_fail(self):
        for output in ['[{"key":"0","score":1}]',
                       '[{"key":"0","score":1},{"key":"0","score":2}]',
                       '[{"key":"0","score":false},{"key":"1","score":2}]']:
            with self.assertRaises(ValueError):
                generate.parse_scores(output, ["0", "1"])
        with self.assertRaises(ValueError):
            generate.parse_relevance("[0] is relevant to the query.", 2)

    def test_original_has_subset_labels_and_four_passes(self):
        data = load_input(self.input)
        client = FakeClient()
        generate.run_original(client, data, self.root)
        output = json.loads((self.root/"generated_game.json").read_text())
        self.assertEqual(set(output["per_subset_relevance"]), {"0", "1", "01"})
        self.assertEqual(len(output["scores_by_pass"]), 4)
        self.assertEqual(output["exact_shapley"], [1.0, 1.0])
        self.assertEqual(len(client.calls), 10)
        self.assertEqual(load_input(self.root/"Shapley_fixture.csv", require_summaries=True)["documents"], data["documents"])

    def test_robustness_full_set_labels_once_common_evaluator(self):
        data = load_input(self.input, require_summaries=True)
        client = FakeClient()
        generate.run_robustness(client, data, self.root)
        calls = client.calls
        self.assertEqual(sum("relevance" in c[0] for c in calls), 1)
        evaluations = [c for c in calls if "evaluation" in c[0]]
        self.assertEqual(len(evaluations), 8)
        self.assertTrue(all(c[1] == generate.REVISION_EVALUATOR for c in evaluations))
        output = json.loads((self.root/"summarizer_robustness.json").read_text())
        self.assertEqual(output["v1"], output["v2"])

    def test_irrelevant_summary_uses_no_generation_call(self):
        client = FakeClient()
        result = generate.summarize(client, ["a", "b"], [False, False], "query", "test", revision=False)
        self.assertEqual(result["summary"], "[0] is not related to the query. [1] is not related to the query.")
        self.assertFalse(client.calls)

    def test_checkpoint_caches_response_and_rejects_different_request(self):
        client = APIClient("openai", self.root, allow_api=True)
        response = {"model": generate.ORIGINAL_MODEL, "choices": [
            {"message": {"content": "fixture"}, "finish_reason": "stop"}]}
        with patch.object(client, "_post", return_value=response) as transport:
            for _ in range(2):
                self.assertEqual(client.chat("one", generate.ORIGINAL_MODEL, "system", "user"), "fixture")
            self.assertEqual(transport.call_count, 1)
            with self.assertRaises(ValueError):
                client.chat("one", generate.ORIGINAL_MODEL, "system", "changed user")
        saved = json.loads((self.root/"api_responses.jsonl").read_text())
        self.assertNotIn("headers", saved)
        self.assertEqual(saved["request_parameters"]["temperature"], 0.1)
        reloaded = APIClient("openai", self.root, allow_api=False)
        self.assertEqual(reloaded.chat("one", generate.ORIGINAL_MODEL, "system", "user"), "fixture")

    def test_embedding_mapping_is_row_aligned_and_numeric(self):
        import numpy as np
        data = load_input(self.input, require_summaries=True)
        client = FakeClient()
        generate.run_embeddings(client, [data], self.root)
        manifest = json.loads((self.root/"embedding_manifest.json").read_text())
        array = np.load(self.root/manifest["files"][data["dataset_name"]], allow_pickle=False)
        self.assertEqual(array.shape, (2, 3072))
        self.assertEqual(client.calls[0][2]["input"], ["One", "Two"])
        self.assertEqual(manifest["input_text"], "singleton summaries")
        self.assertEqual(manifest["records"][data["dataset_name"]]["document_indices"], [0, 1])

    def test_exact_shapley_requires_complete_game(self):
        self.assertEqual(exact_shapley({"0": 1, "1": 2, "01": 3}, 2), [1, 2])
        with self.assertRaises(ValueError):
            exact_shapley({"0": 1}, 2)
        self.assertEqual(len(coalition_keys(10)), 1023)


if __name__ == "__main__":
    unittest.main()
