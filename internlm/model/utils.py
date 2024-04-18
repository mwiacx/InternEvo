from typing import Dict

from internlm.model.modules.mha import MHA


def internlm1_mha_pre_load_convert(model: MHA, state_dict: Dict, prefix: str, *args, **kwargs) -> None:
    if f"{prefix}wqkv.weight" not in state_dict and f"{prefix}Wqkv.weight" in state_dict:
        state_dict[f"{prefix}wqkv.weight"] = state_dict.pop(f"{prefix}Wqkv.weight")

    if f"{prefix}wqkv.bias" not in state_dict and f"{prefix}Wqkv.bias" in state_dict:
        state_dict[f"{prefix}wqkv.bias"] = state_dict.pop(f"{prefix}Wqkv.bias")


def internlm1_mha_save_convert(model: MHA, state_dict: Dict, prefix: str, *args, **kwargs) -> None:
    state_dict[f"{prefix}Wqkv.weight"] = state_dict.pop(f"{prefix}wqkv.weight")

    if f"{prefix}wqkv.bias" in state_dict:
        state_dict[f"{prefix}Wqkv.bias"] = state_dict.pop(f"{prefix}wqkv.bias")
