"""
A simple operator selector, used for compatibility with different platforms such as CUDA and Ascend,
as well as whether to enable flash-attn operator optimization, may be replaced by a more comprehensive
operator compatibility layer in the future.

This file implements support for the roatry embedding operators.
"""

import torch
from einops import rearrange

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import global_context as gpc

try:
    from rotary_emb import apply_rotary as _flash_apply_rotary_func

    flash_rotary_impl = True
except (ModuleNotFoundError, ImportError):
    flash_rotary_impl = False

try:
    from deeplink_ext.internlm_ops import ApplyRotaryEmb as DeeplinkApplyRotaryEmb
    from deeplink_ext.internlm_ops import (
        ApplyRotaryEmbQKV_ as DeeplinkApplyRotaryEmbQKV,
    )

    deeplink_rotary_impl = True
except (ModuleNotFoundError, ImportError):
    deeplink_rotary_impl = False

internlm_accelerator = get_accelerator()


def _torch_apply_rotary_func(
    x1: torch.Tensor,
    x2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    conj: bool = False,
):
    # TODO: improve perfermance.
    assert x1.device == x2.device == cos.device == sin.device, "All inputs must be on the same device"
    assert x1.dtype == x2.dtype == cos.dtype == sin.dtype, "All inputs must have the same dtype"
    assert x1.size() == x2.size(), "Input x1 and x2 must have the same sizes"
    assert cos.size() == sin.size(), "Input cos and sin must have the same sizes"

    x1, x2, cos, sin = x1.float(), x2.float(), cos.float(), sin.float()

    if conj:
        out1.copy_(x1 * cos + x2 * sin)
        out2.copy_(-x1 * sin + x2 * cos)
    else:
        out1.copy_(x1 * cos - x2 * sin)
        out2.copy_(x1 * sin + x2 * cos)

    return out1, out2


def _select_apply_rotary_func(
    x1: torch.Tensor,
    x2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    conj: bool = False,
):
    if gpc.config.model.get("use_flash_attn", False) and flash_rotary_impl:
        _flash_apply_rotary_func(x1, x2, cos, sin, out1, out2, conj)
    else:
        _torch_apply_rotary_func(x1, x2, cos, sin, out1, out2, conj)


# adpated from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py#L35
class ApplyRotaryEmb(torch.autograd.Function):
    """
    ApplyRotaryEmb
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False, in_place: bool = False
    ):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        *_, seqlen, _, head_dim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2

        assert rotary_dim <= head_dim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)

        x_ro = x[..., :rotary_dim]
        x1, x2 = (x_ro[..., ::2], x_ro[..., 1::2]) if interleaved else x_ro.chunk(2, dim=-1)

        if in_place:
            out, o1, o2 = x, x1, x2
        else:
            out = torch.empty_like(x)
            out_ro = out[..., :rotary_dim]
            o1, o2 = (out_ro[..., ::2], out_ro[..., 1::2]) if interleaved else out_ro.chunk(2, dim=-1)

        _select_apply_rotary_func(
            x1, x2, rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d"), o1, o2, False
        )

        if rotary_dim < head_dim and not in_place:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])

        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.in_place = in_place

        return out

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        *_, seqlen, _, head_dim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2

        do_ro = do[..., :rotary_dim]
        do1, do2 = (do_ro[..., ::2], do_ro[..., 1::2]) if ctx.interleaved else do_ro.chunk(2, dim=-1)

        if ctx.in_place:
            dx, dx1, dx2 = do, do1, do2
        else:
            dx = torch.empty_like(do)
            dx_ro = dx[..., :rotary_dim]
            dx1, dx2 = (dx_ro[..., ::2], dx_ro[..., 1::2]) if ctx.interleaved else dx_ro.chunk(2, dim=-1)

        _select_apply_rotary_func(
            do1, do2, rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d"), dx1, dx2, True
        )

        if rotary_dim < head_dim and not ctx.in_place:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])

        return dx, None, None, None, None


class ApplyRotaryEmbQKV(torch.autograd.Function):
    """
    ApplyRotaryEmbQKV_
    """

    @staticmethod
    def forward(
        ctx, qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cos_k=None, sin_k=None, interleaved=False
    ):
        """
            qkv: (batch_size, seqlen, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        _, seqlen, three, _, headdim = qkv.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2

        assert three == 3
        assert seqlen <= rotary_seqlen
        assert rotary_dim <= headdim

        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k

        assert sin.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)

        q_ro = qkv[:, :, 0, :, :rotary_dim]
        q1, q2 = q_ro.chunk(2, dim=-1) if not interleaved else (q_ro[..., ::2], q_ro[..., 1::2])
        re_cos = rearrange(cos[:seqlen], "s d -> s 1 d")
        re_sin = rearrange(sin[:seqlen], "s d -> s 1 d")

        _select_apply_rotary_func(q1, q2, re_cos, re_sin, q1, q2, False)

        k_ro = qkv[:, :, 1, :, :rotary_dim]
        k1, k2 = k_ro.chunk(2, dim=-1) if not interleaved else (k_ro[..., ::2], k_ro[..., 1::2])
        re_cos_k = rearrange(cos_k[:seqlen], "s d -> s 1 d")
        re_sin_k = rearrange(sin_k[:seqlen], "s d -> s 1 d")

        _select_apply_rotary_func(k1, k2, re_cos_k, re_sin_k, k1, k2, False)

        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        ctx.interleaved = interleaved
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        seqlen = dqkv.shape[1]
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2

        dq_ro = dqkv[:, :, 0, :, :rotary_dim]
        dq1, dq2 = dq_ro.chunk(2, dim=-1) if not ctx.interleaved else (dq_ro[..., ::2], dq_ro[..., 1::2])
        re_cos = rearrange(cos[:seqlen], "s d -> s 1 d")
        re_sin = rearrange(sin[:seqlen], "s d -> s 1 d")

        _select_apply_rotary_func(dq1, dq2, re_cos, re_sin, dq1, dq2, True)

        dk_ro = dqkv[:, :, 1, :, :rotary_dim]
        dk1, dk2 = dk_ro.chunk(2, dim=-1) if not ctx.interleaved else (dk_ro[..., ::2], dk_ro[..., 1::2])
        re_cos_k = rearrange(cos_k[:seqlen], "s d -> s 1 d")
        re_sin_k = rearrange(sin_k[:seqlen], "s d -> s 1 d")

        _select_apply_rotary_func(dk1, dk2, re_cos_k, re_sin_k, dk1, dk2, True)

        return dqkv, None, None, None, None, None


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False, in_place: bool = False
):
    # TODO: Support deeplink in a more unified manner
    if internlm_accelerator.get_accelerator_backend() == AcceleratorType.DIPU:
        # TODO: to support in_place argument
        return DeeplinkApplyRotaryEmb.apply(x, cos, sin, interleaved)
    else:
        return ApplyRotaryEmb.apply(x, cos, sin, interleaved, in_place)


def apply_rotary_emb_qkv(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_k=None,
    sin_k=None,
    interleaved=False,
    in_place: bool = False,
):
    assert in_place is True, "in_place = False is not yet implement"

    # TODO: Support deeplink in a more unified manner
    if internlm_accelerator.get_accelerator_backend() == AcceleratorType.DIPU:
        return DeeplinkApplyRotaryEmbQKV.apply(qkv, cos, sin, cos_k, sin_k, interleaved)
    else:
        return ApplyRotaryEmbQKV.apply(qkv, cos, sin, cos_k, sin_k, interleaved)
