"""
xai grok client: openai-compatible chat completions over raw https, stdlib
only, so the grok path adds zero deps (no sdk install, just XAI_API_KEY in
the environment). implements the same two-method interface as
llm.SynthClient, so the pipeline, frontend, and tests don't care which
provider is behind it.

grok supports `temperature`, so text-gen diversity can come from sampling
on top of the prompt-conditioning grid; json calls (taxonomy, judge) never
send temperature so scoring stays as stable as possible.
"""

import json
import os
import random
import time
import urllib.error
import urllib.request

API_URL = "https://api.x.ai/v1/chat/completions"
RETRYABLE = {408, 429, 500, 502, 503, 529}


class GrokClient:
    def __init__(self, model: str, temperature: float | None = None,
                 max_retries: int = 5, api_key: str | None = None):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.api_key = api_key or os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("XAI_API_KEY not set (required for the grok provider)")

    def _chat(self, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        last = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                API_URL, data=data,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as e:
                if e.code not in RETRYABLE:
                    raise
                last = e
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
            time.sleep(min(2 ** attempt + random.random(), 60))
        raise last

    @staticmethod
    def _messages(prompt: str, system: str | None) -> list[dict]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def generate_text(self, prompt: str, system: str | None = None,
                      max_tokens: int = 4096) -> str | None:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": self._messages(prompt, system),
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        choice = self._chat(payload)["choices"][0]
        # empty content covers refusals and filtered outputs; caller skips
        return choice["message"].get("content") or None

    def generate_json(self, prompt: str, schema: dict, system: str | None = None,
                      max_tokens: int = 4096) -> dict | None:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": self._messages(prompt, system),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema, "strict": True},
            },
        }
        choice = self._chat(payload)["choices"][0]
        content = choice["message"].get("content")
        if not content or choice.get("finish_reason") == "length":
            return None
        return json.loads(content)
