"""模型注册表。"""

from __future__ import annotations

from typing import Dict, Type

import torch.nn as nn

_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    def deco(cls: Type[nn.Module]):
        _REGISTRY[name.lower()] = cls
        return cls

    return deco


def get_model_class(name: str) -> Type[nn.Module]:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {list(_REGISTRY)}")
    return _REGISTRY[key]


def list_models() -> list:
    return sorted(_REGISTRY.keys())
