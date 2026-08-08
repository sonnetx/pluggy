"""
hf checkpoint import tests -- no network, no gpus. builds a tiny qwen3, dumps
a reference state dict under HF names into a local safetensors file (using
the module's own writer, so the reader/writer roundtrip is covered too), and
checks the mapping, the tied lm_head, sharded checkpoints, bf16 -> fp32
casting, and the loud failure modes (shape mismatch, unknown keys).

uv run tests/hf_import.py
"""

import json
import shutil
import tempfile

from pathlib import Path

import torch

from pluggy.models.hf_import import (
    hf_to_local, load_hf_weights, read_safetensors, weight_files, write_safetensors,
)
from pluggy.models.qwen3.qwen3 import Qwen3

TINY = {"num_layers": 2, "num_heads": 4, "num_kv_heads": 2,
        "emb_dim": 64, "head_dim": 16, "vocab_size": 100, "ffn_dim": 128}


def hf_state_dict(cfg, tied=True, dtype=torch.bfloat16, seed=0):
    """a random checkpoint under HF qwen3 names, shaped like TINY."""
    g = torch.Generator().manual_seed(seed)
    def t(*shape):
        return torch.randn(*shape, generator=g).to(dtype)
    q = cfg["num_heads"] * cfg["head_dim"]
    kv = cfg["num_kv_heads"] * cfg["head_dim"]
    sd = {"model.embed_tokens.weight": t(cfg["vocab_size"], cfg["emb_dim"]),
          "model.norm.weight": t(cfg["emb_dim"])}
    for i in range(cfg["num_layers"]):
        p = f"model.layers.{i}."
        sd |= {
            p + "input_layernorm.weight": t(cfg["emb_dim"]),
            p + "post_attention_layernorm.weight": t(cfg["emb_dim"]),
            p + "self_attn.q_proj.weight": t(q, cfg["emb_dim"]),
            p + "self_attn.k_proj.weight": t(kv, cfg["emb_dim"]),
            p + "self_attn.v_proj.weight": t(kv, cfg["emb_dim"]),
            p + "self_attn.o_proj.weight": t(cfg["emb_dim"], q),
            p + "self_attn.q_norm.weight": t(cfg["head_dim"]),
            p + "self_attn.k_norm.weight": t(cfg["head_dim"]),
            p + "mlp.gate_proj.weight": t(cfg["ffn_dim"], cfg["emb_dim"]),
            p + "mlp.up_proj.weight": t(cfg["ffn_dim"], cfg["emb_dim"]),
            p + "mlp.down_proj.weight": t(cfg["emb_dim"], cfg["ffn_dim"]),
        }
    if not tied:
        sd["lm_head.weight"] = t(cfg["vocab_size"], cfg["emb_dim"])
    return sd


def test_safetensors_roundtrip(tmp):
    tensors = {"a": torch.randn(3, 5), "b": torch.randn(7).to(torch.bfloat16),
               "c": torch.arange(4, dtype=torch.int64), "scalar": torch.tensor(2.5)}
    path = tmp / "rt.safetensors"
    write_safetensors(tensors, path)
    back = read_safetensors(path)
    assert set(back) == set(tensors)
    for k in tensors:
        assert back[k].dtype == tensors[k].dtype, k
        assert torch.equal(back[k], tensors[k]), k
    print("safetensors roundtrip: ok")


def test_mapping():
    assert hf_to_local("model.embed_tokens.weight") == "token_emb.weight"
    assert hf_to_local("model.layers.3.self_attn.q_norm.weight") == "blocks.3.gqa.q_norm.weight"
    assert hf_to_local("model.layers.27.mlp.down_proj.weight") == "blocks.27.ffn.down_proj.weight"
    assert hf_to_local("model.layers.0.self_attn.rotary_emb.inv_freq") is None
    assert hf_to_local("transformer.h.0.attn.weight") is None
    print("name mapping: ok")


def test_load_tied(tmp):
    ckpt_dir = tmp / "tied"
    ckpt_dir.mkdir()
    sd = hf_state_dict(TINY, tied=True)
    write_safetensors(sd, ckpt_dir / "model.safetensors")

    model = Qwen3(**TINY)
    model.init_weights()
    n = load_hf_weights(model, str(ckpt_dir))
    assert n == len(sd), n
    got = model.state_dict()
    for hf_name, tensor in sd.items():
        local = hf_to_local(hf_name)
        assert got[local].dtype == torch.float32          # bf16 -> fp32 masters
        assert torch.equal(got[local], tensor.to(torch.float32)), local
    # tie preserved: loading the embedding populated the head through the
    # shared tensor, and they are still the same storage
    assert model.lm_head.weight.data_ptr() == model.token_emb.weight.data_ptr()
    assert torch.equal(model.lm_head.weight,
                       sd["model.embed_tokens.weight"].to(torch.float32))
    print("tied load: ok")


def test_load_sharded(tmp):
    ckpt_dir = tmp / "sharded"
    ckpt_dir.mkdir()
    sd = hf_state_dict(TINY, tied=True, seed=1)
    names = sorted(sd)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half],
              "model-00002-of-00002.safetensors": names[half:]}
    weight_map = {}
    for shard, keys in shards.items():
        write_safetensors({k: sd[k] for k in keys}, ckpt_dir / shard)
        weight_map |= dict.fromkeys(keys, shard)
    with open(ckpt_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": weight_map}, f)

    assert len(weight_files(str(ckpt_dir))) == 2
    model = Qwen3(**TINY)
    model.init_weights()
    assert load_hf_weights(model, str(ckpt_dir)) == len(sd)
    assert torch.equal(model.norm.weight, sd["model.norm.weight"].to(torch.float32))
    print("sharded load: ok")


def test_loud_failures(tmp):
    # shape mismatch: model built with the wrong ffn_dim (the exact mistake
    # pluggy's rounded default would make against a hub checkpoint)
    ckpt_dir = tmp / "bad_shape"
    ckpt_dir.mkdir()
    write_safetensors(hf_state_dict(TINY), ckpt_dir / "model.safetensors")
    model = Qwen3(**{**TINY, "ffn_dim": 96})
    model.init_weights()
    try:
        load_hf_weights(model, str(ckpt_dir))
        raise RuntimeError("shape mismatch must raise")
    except AssertionError as e:
        assert "ffn_dim" in str(e)

    # unknown key: not silently dropped
    ckpt_dir = tmp / "bad_key"
    ckpt_dir.mkdir()
    sd = hf_state_dict(TINY) | {"model.layers.0.self_attn.q_proj.bias": torch.zeros(64)}
    write_safetensors(sd, ckpt_dir / "model.safetensors")
    model = Qwen3(**TINY)
    model.init_weights()
    try:
        load_hf_weights(model, str(ckpt_dir))
        raise RuntimeError("unmapped key must raise")
    except AssertionError as e:
        assert "q_proj.bias" in str(e)
    print("loud failures: ok")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="pluggy_hf_import_test_"))
    try:
        test_safetensors_roundtrip(tmp)
        test_mapping()
        test_load_tied(tmp)
        test_load_sharded(tmp)
        test_loud_failures(tmp)
        print("all hf import tests passed")
    finally:
        shutil.rmtree(tmp)
