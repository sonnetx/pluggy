import torch.nn as nn

from pluggy.models.llama3.llama3 import Llama3
from pluggy.models.olmo3.olmo3 import Olmo3
from pluggy.models.qwen3.qwen3 import Qwen3

MODEL_REGISTRY = {
    "qwen3_dense": Qwen3,
    "llama3_2": Llama3,
    "olmo3": Olmo3,
}


def build_model(model_name: str, model_config: dict) -> nn.Module:
    assert model_name in MODEL_REGISTRY.keys(), "model name not in registry"
    return MODEL_REGISTRY[model_name](**model_config)
