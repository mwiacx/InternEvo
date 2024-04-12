from typing import Dict

from internlm.model.modules.mha import MHA


def internlm1_mha_pre_load_convert(model: MHA, state_dict: Dict, *args, **kwargs) -> None:
    if "wqkv.weight" not in state_dict:
        assert "Wqkv.weight" in state_dict, "checkpoint is not compatible, wqkv or Wqkv weight is required."

        state_dict["wqkv.weight"] = state_dict.pop("Wqkv.weight")


def internlm1_mha_save_convert(model: MHA, state_dict: Dict, *args, **kwargs) -> None:
    state_dict["Wqkv.weight"] = state_dict.pop("wqkv.weight")