"""
A simple operator selector, used for compatibility with different platforms such as CUDA and Ascend,
as well as whether to enable flash-attn operator optimization, may be replaced by a more comprehensive
operator compatibility layer in the future.

This file implements support for the linear layer operators.
"""

from typing import Optional, Tuple

import torch
from torch.nn.functional import linear as _torch_linear_forward_op

from internlm.core.context import global_context as gpc

from .utils import OpsBinding

try:
    from fused_dense_lib import linear_bias_wgrad as _flash_linear_backward_op

    flash_attn_impl = True
except (ModuleNotFoundError, ImportError):
    flash_attn_impl = False

_bound_ops = OpsBinding(
    {
        "linear_forward_op": None,
        "linear_backward_op": None,
    }
)


def _select_ops_binding(dtype: torch.dtype, is_cuda: bool = True) -> None:
    dtype_eligible = dtype in (torch.float16, torch.bfloat16) or (
        dtype == torch.float32 and torch.is_autocast_enabled()
    )
    use_flash_attn = gpc.config.model.get("use_flash_attn", False)
    falsh_attn_eligible = flash_attn_impl and dtype_eligible and is_cuda

    if use_flash_attn and falsh_attn_eligible:
        _bound_ops.linear_forward_op = _torch_linear_forward_op
        _bound_ops.linear_backward_op = _flash_linear_backward_op
    else:
        _bound_ops.linear_forward_op = _torch_linear_forward_op
        _bound_ops.linear_backward_op = _linear_bias_wgrad_torch


def _linear_bias_wgrad_torch(_input: torch.Tensor, grad_output: torch.Tensor, has_d_bias: bool):
    assert _input.dtype == grad_output.dtype

    grad_weight = torch.matmul(grad_output.t(), _input)
    grad_bias = grad_output.sum(dim=0) if has_d_bias else None

    return grad_weight, grad_bias


def linear_forward_op(_input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    _is_cuda = _input.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda)
    _select_ops_binding(_input.dtype, _is_cuda)

    return _bound_ops.linear_forward_op(_input, weight, bias)


def linear_backward_op(
    _input: torch.Tensor, weight: torch.Tensor, has_d_bias: bool
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    _is_cuda = _input.is_cuda and weight.is_cuda
    _select_ops_binding(_input.dtype, _is_cuda)

    return _bound_ops.linear_backward_op(_input, weight, has_d_bias)
