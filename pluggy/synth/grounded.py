"""
grounded synthesis: turn uploaded customer documents into pretraining data.
where topic mode generates from the model's own knowledge of a taxonomy,
grounded mode conditions every generation on a chunk of source material and
instructs the model to stay faithful to it (the WRAP/phi-style rephrasing
recipe, and Autodata's requirements-analysis idea applied to a corpus the
customer brings).

chunking is deterministic given the same source files and chunk_words, which
is what makes grounded job keys stable across resume -- the pipeline records
the chunk count in state.json and refuses to resume if it changed.
"""

import json

from pathlib import Path

GROUNDED_MODES = {
    "rephrase": "rewrite the source as clean, well-organized expository prose, "
                "preserving every fact, number, and caveat",
    "textbook": "teach the source's content as a textbook section: add the "
                "background a newcomer needs, define terms, build up to the "
                "source's specifics",
    "qa": "a question-and-answer dialogue that works through the source's "
          "content, several rounds deep, questions a real user would ask",
    "summary": "a dense executive summary of the source followed by an "
               "analysis of its implications and how its pieces relate",
}

_SYSTEM = """\
You write documents for a language model pretraining corpus, grounded in a
source document. Every specific claim (numbers, names, procedures, APIs)
must come from the source; you may add widely-known background, but never
invent specifics the source doesn't support. No meta commentary about the
source or the task. Output only the document."""

_PROMPT = """\
<source>
{chunk}
</source>

Format: {mode_desc}

This is variant {variant} of several documents grounded in this source;
take a distinct angle or emphasis from the obvious one. Write the document
now."""


def load_chunks(src_dir: str, chunk_words: int) -> list[str]:
    """
    read every normalized jsonl in src_dir (rows with a "text" field, as the
    upload endpoint writes) and split docs into word-window chunks. tails
    shorter than a quarter window are dropped as too thin to ground on.
    """
    files = sorted(Path(src_dir).glob("*.jsonl"))
    assert files, (
        f"no jsonl files in {src_dir} -- upload data first (frontend) or drop "
        f"normalized {{'text': ...}} jsonl files there"
    )
    min_words = max(1, chunk_words // 4)
    chunks = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                words = json.loads(line)["text"].split()
                for i in range(0, len(words), chunk_words):
                    part = words[i:i + chunk_words]
                    if len(part) >= min_words:
                        chunks.append(" ".join(part))
    return chunks


def generate_grounded(client, chunk: str, mode: str, variant: int,
                      max_tokens: int) -> str | None:
    text = client.generate_text(
        _PROMPT.format(chunk=chunk, mode_desc=GROUNDED_MODES[mode],
                       variant=variant + 1),
        system=_SYSTEM,
        max_tokens=max_tokens,
    )
    if text is None or len(text.strip()) < 200:
        return None
    return text.strip()
