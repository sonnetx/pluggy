"""
cli entry: uv run -m pluggy.synth.run --config configs/synth_pretrain.json

credentials depend on the config's provider: XAI_API_KEY for grok (stdlib
http, nothing to install), or ANTHROPIC_API_KEY plus the optional dep
(uv pip install -e ".[synth]") for anthropic.
"""

import argparse
import json

from pluggy.synth.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
