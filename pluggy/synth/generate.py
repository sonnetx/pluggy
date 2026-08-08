"""
stage 2: document generation. diversity comes from the conditioning grid
(topic x style x variant index), not from sampling params -- claude-opus-5
rejects temperature/top_p, so each variant gets an explicit "angle" nudge in
the prompt instead.
"""

STYLES = {
    "textbook": "a textbook chapter section: rigorous, defines terms, builds up concepts in order",
    "tutorial": "a hands-on tutorial: walks through doing something concrete, with worked examples",
    "qa": "a question-and-answer dialogue between a curious learner and an expert, several rounds deep",
    "essay": "an analytical essay: takes a position, argues it, addresses counterpoints",
    "reference": "reference documentation: precise, exhaustive on its narrow scope, written to be looked up",
}

_SYSTEM = """\
You write high-quality documents for a language model pretraining corpus.
Write dense, factually careful prose a strong human expert would produce.
No meta commentary, no headers announcing what the document is, no
placeholder text. Output only the document itself."""

_PROMPT = """\
Topic: {topic}
Domain: {domain}
Format: {style_desc}

This is variant {variant} of several documents on this topic; take a
distinct angle from the obvious one (pick an off-distribution but
substantive framing, emphasis, or entry point). Write the document now."""


def generate_doc(client, topic: dict, style: str, variant: int,
                 max_tokens: int) -> str | None:
    text = client.generate_text(
        _PROMPT.format(
            topic=topic["topic"], domain=topic["domain"],
            style_desc=STYLES[style], variant=variant + 1,
        ),
        system=_SYSTEM,
        max_tokens=max_tokens,
    )
    # drop degenerate stubs early so the judge never sees them
    if text is None or len(text.strip()) < 200:
        return None
    return text.strip()
