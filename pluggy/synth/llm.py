"""
thin client over the anthropic sdk. everything else in pluggy/synth talks to
this interface (generate_text / generate_json), so tests can swap in a stub
and no other module imports `anthropic`.

refusals: claude-opus-5 runs safety classifiers that return HTTP 200 with
stop_reason == "refusal" (benign topics occasionally trip them). we opt into
server-side fallbacks by default so a declined request is re-run on the
recommended fallback model inside the same call; if the whole chain refuses
we return None and the pipeline skips that sample instead of crashing.
"""

import json


class SynthClient:
    def __init__(self, model: str, max_retries: int = 5, fallbacks: bool = True):
        # import here so the training stack never needs the dep installed
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=max_retries)
        self.model = model
        self.fallbacks = fallbacks

    def _create(self, *, max_tokens, system=None, messages, output_config=None):
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        if output_config is not None:
            kwargs["output_config"] = output_config
        if self.fallbacks:
            # server-side refusal fallback, routed by refusal category
            return self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **kwargs,
            )
        return self.client.messages.create(**kwargs)

    @staticmethod
    def _text(response):
        return next((b.text for b in response.content if b.type == "text"), None)

    def generate_text(self, prompt: str, system: str | None = None,
                      max_tokens: int = 4096) -> str | None:
        """one text completion. None on refusal (caller skips the sample)."""
        resp = self._create(
            max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            return None
        return self._text(resp)

    def generate_json(self, prompt: str, schema: dict, system: str | None = None,
                      max_tokens: int = 4096) -> dict | None:
        """structured output constrained to `schema`. None on refusal."""
        resp = self._create(
            max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if resp.stop_reason == "refusal":
            return None
        text = self._text(resp)
        # output_config guarantees valid json unless truncated (max_tokens)
        if text is None or resp.stop_reason == "max_tokens":
            return None
        return json.loads(text)
