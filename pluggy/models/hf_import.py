"""
hub checkpoint import for mid-training: load pretrained qwen3 weights into
pluggy's model so a run continues pretraining instead of starting from
random init (config knob: model.init_from_hf).

the safetensors reader/writer is implemented here (the format is 8 bytes of
little-endian header length, a json header mapping tensor name -> {dtype,
shape, data_offsets}, then the raw buffer) rather than depending on the
safetensors package -- same own-the-primitives rule as the collectives.
the writer exists for tests and future checkpoint export.

downloads go through huggingface_hub (already a transitive dep of datasets);
a local directory containing model.safetensors works too, no network.

dtype: hub checkpoints are bf16; weights are copied into the model's own
(fp32) params, which is exactly the mixed-precision master-weight recipe
the trainer expects.
"""

import json
import re

from pathlib import Path

import torch

_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}
_DTYPES_REV = {v: k for k, v in _DTYPES.items()}


def read_safetensors(path) -> dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
        buf = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        # bytearray copy: frombuffer on the read-only bytes would alias
        # non-writable memory and warn on any in-place op downstream
        flat = torch.frombuffer(bytearray(buf[start:end]), dtype=_DTYPES[info["dtype"]])
        out[name] = flat.reshape(info["shape"])
    return out


def write_safetensors(tensors: dict[str, torch.Tensor], path) -> None:
    header, blobs, offset = {}, [], 0
    for name, t in tensors.items():
        t = t.detach().contiguous().cpu()
        data = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
        header[name] = {
            "dtype": _DTYPES_REV[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)


# hf qwen3 name -> pluggy qwen3 name. biases don't exist in either arch, and
# rope frequencies are recomputed, not loaded.
_LAYER_MAP = {
    "input_layernorm.weight": "norm1.weight",
    "post_attention_layernorm.weight": "norm2.weight",
    "self_attn.q_proj.weight": "gqa.q_proj.weight",
    "self_attn.k_proj.weight": "gqa.k_proj.weight",
    "self_attn.v_proj.weight": "gqa.v_proj.weight",
    "self_attn.o_proj.weight": "gqa.o_proj.weight",
    "self_attn.q_norm.weight": "gqa.q_norm.weight",
    "self_attn.k_norm.weight": "gqa.k_norm.weight",
    "mlp.gate_proj.weight": "ffn.gate_proj.weight",
    "mlp.up_proj.weight": "ffn.up_proj.weight",
    "mlp.down_proj.weight": "ffn.down_proj.weight",
}


def hf_to_local(name: str) -> str | None:
    if name == "model.embed_tokens.weight":
        return "token_emb.weight"
    if name == "model.norm.weight":
        return "norm.weight"
    if name == "lm_head.weight":
        return "lm_head.weight"
    m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
    if m and m.group(2) in _LAYER_MAP:
        return f"blocks.{m.group(1)}.{_LAYER_MAP[m.group(2)]}"
    return None


def weight_files(source: str) -> list[Path]:
    """
    resolve `source` (hub repo id, or local dir) to safetensors file paths.
    handles both single-file and index-sharded checkpoints. calling this on
    rank 0 first warms the hub cache so other ranks never race the download.
    """
    local = Path(source)
    if local.is_dir():
        def fetch(fn):
            p = local / fn
            return p if p.exists() else None
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        def fetch(fn):
            try:
                return Path(hf_hub_download(repo_id=source, filename=fn))
            except EntryNotFoundError:
                return None

    index = fetch("model.safetensors.index.json")
    if index is not None:
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        return [fetch(s) for s in shards]
    single = fetch("model.safetensors")
    assert single is not None, (
        f"{source} has neither model.safetensors nor model.safetensors.index.json"
    )
    return [single]


def load_hf_weights(model: torch.nn.Module, source: str) -> int:
    """
    map + copy a hub qwen3 checkpoint into `model` in place (params keep
    their device and dtype; bf16 hub weights land in fp32 master weights).
    returns the number of tensors loaded. loud on anything unexpected:
    unmapped checkpoint keys, shape mismatches, uncovered model params.
    """
    state = {}
    for path in weight_files(source):
        state.update(read_safetensors(path))

    mapped, unmapped = {}, []
    for name, tensor in state.items():
        local = hf_to_local(name)
        (mapped.__setitem__(local, tensor) if local else unmapped.append(name))
    assert not unmapped, (
        f"{source} has keys this importer doesn't map (not a dense qwen3 "
        f"checkpoint?): {unmapped[:8]}"
    )

    model_sd = model.state_dict()
    for name, tensor in mapped.items():
        assert name in model_sd, f"mapped key {name} not in the model"
        assert model_sd[name].shape == tuple(tensor.shape), (
            f"{name}: checkpoint {tuple(tensor.shape)} vs model "
            f"{tuple(model_sd[name].shape)} -- model.config must match the hub "
            f"config exactly (for qwen3 hub checkpoints that includes setting "
            f"ffn_dim to the hub intermediate_size; pluggy's rounded default "
            f"differs)"
        )
    # a tied checkpoint omits lm_head.weight; pluggy ties too, so copying
    # token_emb covers it. anything else missing means a real coverage hole.
    missing = set(model_sd) - set(mapped)
    assert missing <= {"lm_head.weight"}, (
        f"checkpoint leaves model params uninitialized: {sorted(missing)[:8]}"
    )

    with torch.no_grad():
        for name, tensor in mapped.items():
            model_sd[name].copy_(tensor.to(model_sd[name].dtype))
    return len(mapped)
