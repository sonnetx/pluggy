"""
stage 1 of the pipeline: turn a handful of seed domains into a flat topic
list. one llm call per domain, structured output, so the whole taxonomy is a
few cheap calls made once per run (it's checkpointed in state.json and never
regenerated on resume -- topic ids must stay stable for resume to work).
"""

TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {
        "subtopics": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["subtopics"],
    "additionalProperties": False,
}

_PROMPT = """\
You are planning a synthetic pretraining corpus for a language model.

Domain: {domain}

Propose {n} specific, diverse subtopics inside this domain. Each subtopic
should be concrete enough to write a full standalone document about (not a
one-word label), and the set should cover the domain broadly: mix
introductory and advanced material, theory and application, common and
under-documented corners. Return only the subtopic list."""


def build_taxonomy(client, domains: list[str], subtopics_per_domain: int) -> list[dict]:
    """returns [{"id": int, "domain": str, "topic": str}, ...]"""
    topics = []
    for domain in domains:
        out = client.generate_json(
            _PROMPT.format(domain=domain, n=subtopics_per_domain),
            TAXONOMY_SCHEMA,
        )
        if out is None:
            print(f"taxonomy: skipped domain {domain!r} (refused)")
            continue
        for sub in out["subtopics"]:
            topics.append({"id": len(topics), "domain": domain, "topic": sub})
    return topics
