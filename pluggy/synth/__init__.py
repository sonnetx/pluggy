"""
synthetic pretraining data pipeline (agentic generate -> judge -> refine loop,
after Autodata, arXiv:2606.25996). produces jsonl shards with a "text" field
that stream straight into pluggy's existing dataloader.

requires the optional `anthropic` dep: uv pip install -e ".[synth]"
"""
