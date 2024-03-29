# Copyright (c) OpenMMLab. All rights reserved.
import types

from contextlib import contextmanager
from typing import Any, Callable, Tuple
from enum import Enum
from functools import update_wrapper

import torch

from internlm.utils.logger import get_logger

logger = get_logger(__file__)


@contextmanager
def read_base():
    """Context manager to mark the base config.

    The pure Python-style configuration file allows you to use the import
    syntax. However, it is important to note that you need to import the base
    configuration file within the context of ``read_base``, and import other
    dependencies outside of it.

    You can see more usage of Python-style configuration in the `tutorial`_

    .. _tutorial: https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta  # pylint: disable=line-too-long
    """  # noqa: E501
    yield


class QKVPackType(Enum):
    QKVPACKED = 2
    KVPACKED = 3
    QKVSPLITED = 4

class CuSeqlenType(Enum):
    With = True
    WithOut = False

def check_attention_argument(*args, **kwargs) -> str:
    # self, qkv, ...
    # self, q, kv, ....
    # self, q, k, v, ...
    # self, qkv, cu_seqlens, max_seqlen, ...
    # self, q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...
    # self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...
    def __qkv_checker(num_args: int):
        if num_args < 2:
            return "qkv" in kwargs
        else:
            # qkv: [batch, seqlen, 3, n_head, headdim]
            return args[1].shape == 5

    def __kv_checker(num_args: int):
        if num_args < 3:
            return "kv" in kwargs
        else:
            # kv: [batch, seqlen, 3, n_head, headdim]
            return args[2].shape == 5

    def __cu_seqlens_checker(num_args: int, check_idx: int):
        if num_args < (check_idx + 1):
            return "cu_seqlens" in kwargs or "cu_seqlens_q" in kwargs
        else:
            return isinstance(num_args[check_idx], torch.Tensor)

    if __qkv_checker(len(args)):
        # qkv packed, and we should check cu_seqlens with index 2
        qkv_pack_type = QKVPackType.QKVPACKED
    elif __kv_checker(len(args)):
        # kv packed, and we should check cu_seqlens with index 3
        qkv_pack_type = QKVPackType.KVPACKED
    else:
        # qkv splited, and we should check cu_seqlens with index 4
        qkv_pack_type = QKVPackType.QKVSPLITED

    with_cu_seqlens = __cu_seqlens_checker(len(args), int(qkv_pack_type))

    return str(qkv_pack_type), str(with_cu_seqlens)


def _params_dispatch_with_condition(func: Callable, condition: Callable):

    registry = {}
    funcname = getattr(func, "__name__", "singledispatch function")

    def dispatch(_type: str) -> Callable:
        try:
            impl = registry[_type]
        except KeyError:
            logger.error("unknown dispatch type: %s", _type)

        return impl

    def register(conditions: Tuple[str], func: Callable) -> None:
        _type = "-".join(conditions)

        assert _type not in registry, f"Repeatedly register dispatch functions for pattern {_type}"

        registry[_type] = func

    def wrapper(*args, **kwargs):
        if not args:
            raise TypeError(f"{funcname} requires at least " "1 positional argument")

        _type = "-".join(condition(*args, **kwargs))

        return dispatch(_type)(*args, **kwargs)

    registry[""] = func
    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = types.MappingProxyType(registry)
    update_wrapper(wrapper, func)
    return wrapper

def params_dispatch_with_condition(condition: Callable):

    def decorator_wrapper(func: Callable):
        return _params_dispatch_with_condition(func, condition)

    return decorator_wrapper
