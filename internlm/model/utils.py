from typing import Dict

from internlm.model.modules.mha import MHA


def internlm1_mha_pre_load_convert(model: MHA, state_dict: Dict,  prefix: str, *args, **kwargs) -> None:
    if f"{prefix}wqkv.weight" not in state_dict:
        assert f"{prefix}Wqkv.weight" in state_dict, "checkpoint is not compatible, wqkv or Wqkv weight is required."

        state_dict[f"{prefix}wqkv.weight"] = state_dict.pop(f"{prefix}Wqkv.weight")


def internlm1_mha_save_convert(model: MHA, state_dict: Dict, prefix: str, *args, **kwargs) -> None:

    state_dict[f"{prefix}Wqkv.weight"] = state_dict.pop(f"{prefix}wqkv.weight")
