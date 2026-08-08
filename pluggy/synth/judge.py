"""
stages 3 and 4: evaluation and refinement. a separate judge call scores each
document against a fixed rubric (structured output, integer 1..10); docs at
or above `min_score` pass, docs in the refine band get one rewrite that
feeds the judge's weaknesses back to the generator, everything else drops.

the judge schema keeps `score` an untyped integer because structured outputs
don't support minimum/maximum; the rubric prompt pins the range instead.
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "weaknesses": {"type": "string"},
    },
    "required": ["score", "weaknesses"],
    "additionalProperties": False,
}

_JUDGE_PROMPT = """\
Score this document as pretraining data for a language model, on an integer
scale from 1 to 10:

- factual accuracy and internal consistency (errors are disqualifying)
- density: how much the document teaches per token, no filler or repetition
- depth: goes beyond surface-level summary of the topic
- writing quality: clear, natural expert prose

8-10: strong expert material. 5-7: usable but flawed. 1-4: inaccurate,
padded, or vacuous. Also state the main weaknesses in one or two sentences.

<document>
{doc}
</document>"""

# grounded docs get an extra, dominant criterion: faithfulness to the source
# they were synthesized from. the judge sees both, so hallucinated specifics
# are checkable rather than merely plausible.
_JUDGE_GROUNDED_PROMPT = """\
Score this document as pretraining data for a language model, on an integer
scale from 1 to 10. It was synthesized from the source below; judge it on:

- faithfulness: every specific claim (numbers, names, procedures) is
  supported by the source; invented specifics are disqualifying
- coverage and density: uses the source's substance well, no filler
- writing quality: clear, natural expert prose

8-10: faithful and strong. 5-7: usable but flawed. 1-4: contradicts or
fabricates beyond the source, or is padded/vacuous. Also state the main
weaknesses in one or two sentences.

<source>
{source}
</source>

<document>
{doc}
</document>"""

_REFINE_PROMPT = """\
Rewrite the document below to fix these weaknesses, keeping its topic,
format, and angle. Output only the rewritten document.

Weaknesses: {weaknesses}

<document>
{doc}
</document>"""


def judge_doc(client, doc: str, source: str | None = None) -> dict | None:
    prompt = (_JUDGE_GROUNDED_PROMPT.format(doc=doc, source=source)
              if source is not None else _JUDGE_PROMPT.format(doc=doc))
    out = client.generate_json(prompt, JUDGE_SCHEMA, max_tokens=1024)
    if out is None or not 1 <= out["score"] <= 10:
        return None
    return out


def refine_doc(client, doc: str, weaknesses: str, max_tokens: int) -> str | None:
    text = client.generate_text(
        _REFINE_PROMPT.format(doc=doc, weaknesses=weaknesses),
        max_tokens=max_tokens,
    )
    if text is None or len(text.strip()) < 200:
        return None
    return text.strip()


def evaluate(client, doc: str, quality_cfg: dict, max_tokens: int,
             source: str | None = None):
    """
    full judge -> maybe-refine -> re-judge loop for one doc. pass `source`
    for grounded docs so the judge scores faithfulness against it.
    returns (final_doc, score) or (None, None) if the doc should drop.
    """
    min_score = quality_cfg.get("min_score", 7)
    refine_min = quality_cfg.get("refine_min", 5)
    rounds = quality_cfg.get("max_refine_rounds", 1)

    verdict = judge_doc(client, doc, source)
    if verdict is None:
        return None, None
    for _ in range(rounds):
        if verdict["score"] >= min_score:
            break
        if verdict["score"] < refine_min:
            return None, None
        better = refine_doc(client, doc, verdict["weaknesses"], max_tokens)
        if better is None:
            return None, None
        doc = better
        verdict = judge_doc(client, doc, source)
        if verdict is None:
            return None, None
    if verdict["score"] < min_score:
        return None, None
    return doc, verdict["score"]
