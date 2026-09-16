"""Minimal opt-in HTTP client with resumable, credential-free response journals."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import urllib.error
import urllib.request


ENDPOINTS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class APIError(RuntimeError):
    """A sanitized error that never contains headers, tokens, or server bodies."""


class APIClient:
    def __init__(self, provider, output_dir, *, allow_api=False, timeout=180):
        if provider not in ENDPOINTS:
            raise ValueError("Unsupported provider")
        self.provider = provider
        self.allow_api = allow_api
        self.timeout = timeout
        self.journal = Path(output_dir) / "api_responses.jsonl"
        self.cache = {}
        if self.journal.exists():
            for line in self.journal.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["call_id"] in self.cache:
                    raise ValueError("Duplicate call identifiers in response journal")
                self.cache[row["call_id"]] = row

    def _post(self, path, body):
        if not self.allow_api:
            raise APIError("Network access requires explicit --allow-api")
        base, environment_name = ENDPOINTS[self.provider]
        # Read only the selected credential at call time. Never write or log it.
        credential = os.environ.get(environment_name)
        if not credential:
            raise APIError(f"Selected provider credential is absent: {environment_name}")
        request = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": "Bearer " + credential,
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise APIError(f"{self.provider} HTTP {error.code}; response body omitted") from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            raise APIError(f"{self.provider} transport or response-decoding failure; details omitted") from None

    def call(self, call_id, path, body):
        request_hash = digest({"provider": self.provider, "path": path, "body": body})
        if call_id in self.cache:
            row = self.cache[call_id]
            if row["request_sha256"] != request_hash:
                raise ValueError("A cached call has different inputs; choose a new output directory")
            return row["response"]
        raw = self._post(path, body)
        # Allowlist returned metadata and model output. Never persist a request/header.
        saved = {key: raw[key] for key in
                 ("id", "model", "created", "usage", "system_fingerprint", "provider") if key in raw}
        if path == "/embeddings":
            saved["data"] = [{"index": row["index"], "embedding": row["embedding"]}
                             for row in raw.get("data", [])]
        else:
            choices = raw.get("choices", [])
            if not choices:
                raise APIError("Provider returned no completion choices; details omitted")
            saved["content"] = choices[0].get("message", {}).get("content")
            saved["finish_reason"] = choices[0].get("finish_reason")
        row = {"call_id": call_id, "collected_at_utc": now(), "provider": self.provider,
               "endpoint": ENDPOINTS[self.provider][0] + path,
               "request_sha256": request_hash,
               "request_parameters": {key: body[key] for key in
                                      ("model", "temperature", "max_tokens", "dimensions",
                                       "encoding_format", "response_format") if key in body},
               "response": saved}
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.cache[call_id] = row
        return saved

    def chat(self, call_id, model, system, user, schema=None, json_mode=False):
        body = {"model": model, "messages": [{"role": "system", "content": system},
                                               {"role": "user", "content": user}],
                "temperature": 0.1}
        if self.provider == "openrouter":
            body["max_tokens"] = 32768
            if json_mode:
                body["response_format"] = {"type": "json_object"}
        elif schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "summary_response" if "summary" in schema["properties"] else "evaluation_response",
                "strict": True, "schema": schema}}
        result = self.call(call_id, "/chat/completions", body)
        if result.get("finish_reason") != "stop" or not isinstance(result.get("content"), str):
            raise APIError("Incomplete or refused completion; raw model output is retained in the response journal")
        return result["content"]
