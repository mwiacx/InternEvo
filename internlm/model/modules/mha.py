#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from internlm.core.context import global_context as gpc
from internlm.model.modules.embedding import new_rotary_embedding
from internlm.model.modules.linear import new_linear
from internlm.model.modules.utils import update_kv_cache
from internlm.model.ops.attention import CrossAttention, SelfAttention

# MHA, qkvpacked, inference
# def _inference(self, x, inference_params=None, **kwargs):
#         qkv = self.Wqkv(x)
#         qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)

#         bsz, *_ = qkv.shape  # 是否需要支持 data-packed

#         if self.use_dynamic_ntk_rope:
#             q = qkv[:, :, 0]
#             assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
#             kv = update_kv_cache(qkv[:, :, 1:], inference_params, self.layer_idx)
#             if inference_params.sequence_len_offset != 0:
#                 # q shape: [bsz, 1, nheads, head_dim]
#                 # kv shape: [bsz, seqlen, 2, nheads, head_dim]
#                 # bsz, seq_len, _, nheads, head_dim = kv.shape
#                 # q = torch.cat([q.new_zeros(size=(bsz, seq_len - 1, nheads, head_dim)), q], dim=1).unsqueeze(2)

#                 # qkv = torch.cat([q, kv], dim=2)
#                 if self.rotary_emb_dim > 0:
#                     k = kv[:, :, 1].squeeze(2)
#                     q = self.rotary_emb(
#                         q.squeeze(2), offsets=inference_params.sequence_len_offset, cache_type="query"
#                     ).unsqueeze(2)
#                     self.rotary_emb(k, offsets=0, cache_type="key", in_place=True)  # in-place
#                     # qkv = self.rotary_emb(qkv)
#                 # q = qkv[:, [-1], 0]
#                 # kv = qkv[:, :, 1:]
#             else:
#                 if inference_params.sequence_len_offset > self.max_position_embeddings:
#                     warnings.warn(
#                         "Notice your prompt's length is longer than model's max_position_embeddings: "
#                         f"{self.max_position_embeddings}, which will cause deviations in dynamic ntk calculations."
#                     )
#                 if self.rotary_emb_dim > 0:
#                     # kwargs["inference_params"] = inference_params
#                     # qkv = self.rotary_emb(qkv, **kwargs)
#                     k = qkv[:, :, 1].squeeze(2)
#                     q = self.rotary_emb(q.squeeze(2), offsets=0, cache_type="query").unsqueeze(2)
#                     self.rotary_emb(k, offsets=0, cache_type="key", in_place=True)  # in-place
#                     # q = qkv[:, :, 0]
#                     kv = qkv[:, :, 1:]
#         else:
#             assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
#             q, k, v = (x.squeeze(2) for x in qkv.chunk(chunks=3, dim=2))
#             kv = torch.stack([k, v], dim=2)
#             assert self.rotary_emb_dim > 0, "You should use rotary_emb."
#             cu_seqlens = kwargs.get("cu_seqlens", None)
#             # max_seqlen = kwargs.get("max_seqlen", None)

#             if hasattr(inference_params, "attention_mask") and inference_params.attention_mask is not None:
#                 empties = inference_params.attention_mask[..., -1].sum(dim=-1)
#                 if inference_params.sequence_len_offset == 0:
#                     q = self.rotary_emb(
#                         q, offsets=0, cache_type="query", left_padding_mask=inference_params.attention_mask
#                     )
#                     k = self.rotary_emb(
#                         k, offsets=0, cache_type="key", left_padding_mask=inference_params.attention_mask
#                     )
#                 else:
#                     if inference_params.sequence_len_offset > self.max_position_embeddings:
#                         warnings.warn(
#                             "Notice your prompt's length is longer than model's max_position_embeddings: "
#                             f"{self.max_position_embeddings}, may cause deviations in dynamic ntk calculations."
#                         )
#                     q = q.squeeze(1)
#                     k = k.squeeze(1)
#                     q = self.rotary_emb(
#                         q,
#                         inference_params.sequence_len_offset * torch.ones(q.size(0), dtype=torch.int, device=q.device)
#                         - empties,
#                         cache_type="query",
#                     ).unsqueeze(1)
#                     k = self.rotary_emb(
#                         k,
#                         inference_params.sequence_len_offset * torch.ones(k.size(0), dtype=torch.int, device=k.device)
#                         - empties,
#                         cache_type="key",
#                     ).unsqueeze(1)
#             else:
#                 q = self.rotary_emb(q, inference_params.sequence_len_offset, cache_type="query")
#                 k = self.rotary_emb(k, inference_params.sequence_len_offset, cache_type="key")

#             kv = torch.stack([k, v], dim=2)
#             kv = update_kv_cache(kv, inference_params, self.layer_idx)

#         if hasattr(inference_params, "attention_mask") and inference_params.attention_mask is not None:
#             if inference_params.sequence_len_offset == 0:  # First entrance, attnmask (bs*seqlen*seqlen)
#                 attn_mask = inference_params.attention_mask[:, None, ...]
#                 attn_mask = torch.logical_or(torch.ones_like(attn_mask, dtype=torch.bool).triu(diagonal=1), attn_mask)
#                 attn_mask4flsh = ~attn_mask[:, :, -1, :].view(bsz, -1)
#                 cu_seqlens = torch.concat(
#                     [
#                         torch.tensor([0], dtype=torch.int32, device=attn_mask4flsh.device),
#                         attn_mask4flsh.sum(dim=-1).to(dtype=torch.int32),
#                     ],
#                     dim=0,
#                 )
#                 cu_seqlens = cu_seqlens.cumsum(dim=0, dtype=torch.int32)
#                 max_seqlen_q = attn_mask4flsh.shape[-1]
#                 max_seqlen_k = attn_mask4flsh.shape[-1]
#                 total_q = q.masked_select(attn_mask4flsh.view(bsz, -1, 1, 1)).view(-1, q.shape[-2], q.shape[-1])
#                 total_kv = kv.masked_select(attn_mask4flsh.view(bsz, -1, 1, 1, 1)).view(
#                     -1, kv.shape[-3], kv.shape[-2], kv.shape[-1]
#                 )

#                 # if gpc.config.model.dtype is torch.float32 and gpc.config.model.use_flash_attn:
#                 #     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#                 #         if total_q.dtype not in [torch.float16, torch.bfloat16]:
#                 #             total_q = total_q.to(torch.bfloat16)
#                 #         if total_kv.dtype not in [torch.float16, torch.bfloat16]:
#                 #             total_kv = total_kv.to(torch.bfloat16)

#                 # if gpc.config.model.use_flash_attn:
#                 #     try:
#                 #         from flash_attn.flash_attn_interface import (
#                 #             flash_attn_unpadded_func,
#                 #         )
#                 #     except ImportError:
#                 #         try:
#                 #             from flash_attn.flash_attn_interface import (
#                 #                 flash_attn_unpadded_kvpacked_func as flash_attn_unpadded_func,
#                 #             )
#                 #         except ImportError:
#                 #             try:
#                 #                 from flash_attn.flash_attn_interface import (
#                 #                     flash_attn_varlen_kvpacked_func as flash_attn_unpadded_func,
#                 #                 )
#                 #             except ImportError:
#                 #                 raise ImportError("Please check your flash_attn version >= 1.0.5.")

#                 output = self.inner_attn(
#                     total_q,
#                     total_kv,
#                     cu_seqlens,
#                     cu_seqlens,
#                     max_seqlen_q,
#                     max_seqlen_k,
#                     0.0,
#                     None,
#                     True,
#                     False,
#                 ).to(x.dtype)
#                 # else:
#                 #     attn_scores = torch.matmul(total_q, total_kv.transpose(-2, -1)) / (cu_seqlens**0.5)
#                 #     attn_weights = F.softmax(attn_scores, dim=-1)
#                 #     output = torch.matmul(attn_weights, total_kv)

#                 context = torch.zeros_like(q)
#                 context = context.masked_scatter_(attn_mask4flsh.view(bsz, -1, 1, 1), output)

#             else:
#                 attn_mask = inference_params.attention_mask[:, -1, :].view(bsz, 1, 1, -1)

#                 k, v = torch.chunk(kv, 2, dim=2)
#                 k = k.squeeze(2)
#                 v = v.squeeze(2)
#                 sp = k.shape
#                 scores = torch.einsum(
#                     "blhd,bnhd->bhln",
#                     q,
#                     k.reshape(sp[0], sp[1], q.size(2), sp[3]),
#                 ) / math.sqrt(q.size(-1))
#                 scores = scores.masked_fill(attn_mask, -65000.0)
#                 scores = F.softmax(scores, dim=-1)  # bsz x h x L x L
#                 context = torch.einsum(
#                     "bhmn,bnhd->bmhd",
#                     scores,
#                     v.reshape(sp[0], sp[1], q.size(2), sp[3]),
#                 )
#         else:
#             context = self.inner_cross_attn(q, kv, causal=True)

#         context = rearrange(context, "... h d -> ... (h d)")

#         out = self.out_proj(context)
#         return out


# GQA, qkv splited, inference, w/o wqkv and wo
# assert self.rotary_emb_dim > 0
#             if hasattr(inference_params, "attention_mask") and inference_params.attention_mask is not None:
#                 empties = inference_params.attention_mask[..., -1].sum(dim=-1)
#                 if inference_params.sequence_len_offset == 0:
#                     q = self.rotary_emb(
#                         q, offsets=0, cache_type="query", left_padding_mask=inference_params.attention_mask
#                     )
#                     k = self.rotary_emb(
#                         k, offsets=0, cache_type="key", left_padding_mask=inference_params.attention_mask
#                     )
#                 else:
#                     q = q.squeeze(1)
#                     k = k.squeeze(1)
#                     q = self.rotary_emb(
#                         q,
#                         inference_params.sequence_len_offset * torch.ones(q.size(0), dtype=torch.int, device=q.device)
#                         - empties,
#                         cache_type="query",
#                     ).unsqueeze(1)
#                     k = self.rotary_emb(
#                         k,
#                         inference_params.sequence_len_offset * torch.ones(k.size(0), dtype=torch.int, device=k.device)
#                         - empties,
#                         cache_type="key",
#                     ).unsqueeze(1)
#             else:
#                 raise NotImplementedError(
#                     "You should make sure you are aware that you are changing the method of generating."
#                     "According to your generation function instead of inference/seq_generator_module.py, "
#                     "You may implement here for normal running."
#                 )

#             kv = torch.stack([k, v], dim=2)

#             assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
#             if hasattr(inference_params, "window_size") and inference_params.window_size is not None:
#                 if inference_params.window_size <= inference_params.sequence_len_offset:
#                     assert kv.size(1) == 1, "update kv lenth more than 1"
#                     inference_params.key_value_memory_dict[self.layer_idx][
#                         :, inference_params.keep_first : inference_params.window_size - 1, ...
#                     ] = inference_params.key_value_memory_dict[self.layer_idx][
#                         :, -(inference_params.window_size - 1 - inference_params.keep_first) :, ...
#                     ].clone()
#                     inference_params.real_sequence_len_offset = inference_params.sequence_len_offset
#                     inference_params.sequence_len_offset = inference_params.window_size - 1

#                     kv = update_kv_cache(kv, inference_params, self.layer_idx)

#                     inference_params.sequence_len_offset = inference_params.real_sequence_len_offset
#                 else:
#                     kv = update_kv_cache(kv, inference_params, self.layer_idx)
#             else:
#                 kv = update_kv_cache(kv, inference_params, self.layer_idx)

#             # When using FP16, there is a high probability of NAN in the KV.
#             # Since NAN cannot be removed by multiplying with and 0, it needs
#             # to be removed manually here.
#             kv = torch.where(torch.isnan(kv), 0, kv)

#             if hasattr(inference_params, "attention_mask") and inference_params.attention_mask is not None:
#                 assert self.use_flash_attn is True
#                 from flash_attn.flash_attn_interface import FlashAttnVarlenKVPackedFunc

#                 if inference_params.sequence_len_offset == 0:  # First entrance, attnmask (bs*seqlen*seqlen)
#                     attn_mask = inference_params.attention_mask[:, None, ...]
#                     attn_mask = torch.logical_or(
#                         torch.ones_like(attn_mask, dtype=torch.bool).triu(diagonal=1), attn_mask
#                     )
#                     attn_mask4flsh = ~attn_mask[:, :, -1, :].view(bsz, -1)
#                     cu_seqlens = torch.concat(
#                         [
#                             torch.tensor([0], dtype=torch.int32, device=attn_mask4flsh.device),
#                             attn_mask4flsh.sum(dim=-1).to(dtype=torch.int32),
#                         ],
#                         dim=0,
#                     )
#                     cu_seqlens = cu_seqlens.cumsum(dim=0, dtype=torch.int32)
#                     max_seqlen_q = attn_mask4flsh.shape[-1]
#                     max_seqlen_k = attn_mask4flsh.shape[-1]
#                     total_q = q.masked_select(attn_mask4flsh.view(bsz, -1, 1, 1)).view(-1, q.shape[-2], q.shape[-1])
#                     total_kv = kv.masked_select(attn_mask4flsh.view(bsz, -1, 1, 1, 1)).view(
#                         -1, kv.shape[-3], kv.shape[-2], kv.shape[-1]
#                     )

#                     # if self.dtype is torch.float32:
#                     #     if total_q.dtype not in [torch.float16, torch.bfloat16]:
#                     #         total_q = total_q.to(torch.bfloat16)
#                     #     if total_kv.dtype not in [torch.float16, torch.bfloat16]:
#                     #         total_kv = total_kv.to(torch.bfloat16)
#                     #     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#                     #         output = FlashAttnVarlenKVPackedFunc.apply(
#                     #             total_q,
#                     #             total_kv,
#                     #             cu_seqlens,
#                     #             cu_seqlens,
#                     #             max_seqlen_q,
#                     #             max_seqlen_k,
#                     #             0.0,
#                     #             None,
#                     #             True,
#                     #             False,
#                     #         ).to(self.dtype)
#                     # else:
#                     output = FlashAttnVarlenKVPackedFunc.apply(
#                         total_q,
#                         total_kv,
#                         cu_seqlens,
#                         cu_seqlens,
#                         max_seqlen_q,
#                         max_seqlen_k,
#                         0.0,
#                         None,
#                         True,
#                         False,
#                     )

#                     context = torch.zeros_like(q)
#                     context = context.masked_scatter_(attn_mask4flsh.view(bsz, -1, 1, 1), output)

#                 else:
#                     attn_mask = inference_params.attention_mask[:, -1, :].view(bsz, 1, 1, -1)
#                     if hasattr(inference_params, "window_size") and inference_params.window_size is not None:
#                         if inference_params.window_size <= inference_params.sequence_len_offset:
#                             attn_mask = torch.concat(
#                                 [
#                                     attn_mask[..., : inference_params.keep_first],
#                                     attn_mask[..., -(inference_params.window_size - inference_params.keep_first) :],
#                                 ],
#                                 dim=-1,
#                             )

#                     k, v = torch.chunk(kv, 2, dim=2)
#                     k = k.squeeze(2)
#                     v = v.squeeze(2)
#                     sp = k.shape
#                     expansion = q.size(2) // k.size(2)
#                     scores = torch.einsum(
#                         "blhd,bnhd->bhln",
#                         q,
#                         k.unsqueeze(3).expand(-1, -1, -1, expansion, -1).reshape(sp[0], sp[1], q.size(2), sp[3]),
#                     ) / math.sqrt(q.size(-1))
#                     scores = scores.masked_fill(attn_mask, -65000.0)
#                     scores = F.softmax(scores, dim=-1)  # bsz x h x L x L
#                     context = torch.einsum(
#                         "bhmn,bnhd->bmhd",
#                         scores,
#                         v.unsqueeze(3).expand(-1, -1, -1, expansion, -1).reshape(sp[0], sp[1], q.size(2), sp[3]),
#                     )
#             else:
#                 # if self.dtype is torch.float32 and self.use_flash_attn:
#                 #     if q.dtype not in [torch.float16, torch.bfloat16]:
#                 #         q = q.to(torch.bfloat16)
#                 #     if kv.dtype not in [torch.float16, torch.bfloat16]:
#                 #         kv = kv.to(torch.bfloat16)
#                 #     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#                 #         context = self.inner_cross_attn(q, kv, causal=True).to(self.dtype)
#                 # else:
#                 context = self.inner_cross_attn(q, kv, causal=True)


# GQA, qkv packed, inference, w/o wqkv and wo
# assert self.rotary_emb_dim > 0
#         if hasattr(inference_params, "attention_mask") and inference_params.attention_mask is not None:
#             empties = inference_params.attention_mask[..., -1].sum(dim=-1)
#             if inference_params.sequence_len_offset == 0:
#                 q = self.rotary_emb(
#                     q, offsets=0, cache_type="query", left_padding_mask=inference_params.attention_mask
#                 )
#                 k = self.rotary_emb(
#                     k, offsets=0, cache_type="key", left_padding_mask=inference_params.attention_mask
#                 )
#             else:
#                 q = q.squeeze(1)
#                 k = k.squeeze(1)
#                 q = self.rotary_emb(
#                     q,
#                     inference_params.sequence_len_offset * torch.ones(q.size(0), dtype=torch.int, device=q.device)
#                     - empties,
#                     cache_type="query",
#                 ).unsqueeze(1)
#                 k = self.rotary_emb(
#                     k,
#                     inference_params.sequence_len_offset * torch.ones(k.size(0), dtype=torch.int, device=k.device)
#                     - empties,
#                     cache_type="key",
#                 ).unsqueeze(1)
#         else:
#             raise NotImplementedError(
#                 "You should make sure you are aware that you are changing the method of generating."
#                 "According to your generation function instead of inference/seq_generator_module.py, "
#                 "You may implement here for normal running."
#             )

#         kv = torch.stack([k, v], dim=2)

#         assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
#         if hasattr(inference_params, "window_size") and inference_params.window_size is not None:
#             if inference_params.window_size <= inference_params.sequence_len_offset:
#                 assert kv.size(1) == 1, "update kv lenth more than 1"
#                 inference_params.key_value_memory_dict[self.layer_idx][
#                     :, inference_params.keep_first : inference_params.window_size - 1, ...
#                 ] = inference_params.key_value_memory_dict[self.layer_idx][
#                     :, -(inference_params.window_size - 1 - inference_params.keep_first) :, ...
#                 ].clone()
#                 inference_params.real_sequence_len_offset = inference_params.sequence_len_offset
#                 inference_params.sequence_len_offset = inference_params.window_size - 1

#                 kv = update_kv_cache(kv, inference_params, self.layer_idx)

#                 inference_params.sequence_len_offset = inference_params.real_sequence_len_offset
#             else:
#                 kv = update_kv_cache(kv, inference_params, self.layer_idx)
#         else:
#             kv = update_kv_cache(kv, inference_params, self.layer_idx)

#         # When using FP16, there is a high probability of NAN in the KV.
#         # Since NAN cannot be removed by multiplying with and 0, it needs
#         # to be removed manually here.
#         kv = torch.where(torch.isnan(kv), 0, kv)

#         if hasattr(inference_params, "attention_mask") and inference_params.attention_mask is not None:
#             assert self.use_flash_attn is True
#             from flash_attn import flash_attn_varlen_kvpacked_func

#             if inference_params.sequence_len_offset == 0:  # First entrance, attnmask (bs*seqlen*seqlen)
#                 attn_mask = inference_params.attention_mask[:, None, ...]
#                 attn_mask = torch.logical_or(
#                     torch.ones_like(attn_mask, dtype=torch.bool).triu(diagonal=1), attn_mask
#                 )
#                 attn_mask4flsh = ~attn_mask[:, :, -1, :].view(bsz, -1)
#                 cu_seqlens = torch.concat(
#                     [
#                         torch.tensor([0], dtype=torch.int32, device=attn_mask4flsh.device),
#                         attn_mask4flsh.sum(dim=-1).to(dtype=torch.int32),
#                     ],
#                     dim=0,
#                 )
#                 cu_seqlens = cu_seqlens.cumsum(dim=0, dtype=torch.int32)
#                 max_seqlen_q = attn_mask4flsh.shape[-1]
#                 max_seqlen_k = attn_mask4flsh.shape[-1]
#                 total_q = q.masked_select(attn_mask4flsh.view(bsz, -1, 1, 1)).view(-1, q.shape[-2], q.shape[-1])
#                 total_kv = kv.masked_select(attn_mask4flsh.view(bsz, -1, 1, 1, 1)).view(
#                     -1, kv.shape[-3], kv.shape[-2], kv.shape[-1]
#                 )
#                 if self.dtype is torch.float32:
#                     if total_q.dtype not in [torch.float16, torch.bfloat16]:
#                         total_q = total_q.to(torch.bfloat16)
#                     if total_kv.dtype not in [torch.float16, torch.bfloat16]:
#                         total_kv = total_kv.to(torch.bfloat16)
#                     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#                         output = flash_attn_varlen_kvpacked_func(
#                             q=total_q,
#                             kv=total_kv,
#                             cu_seqlens_q=cu_seqlens,
#                             cu_seqlens_k=cu_seqlens,
#                             max_seqlen_q=max_seqlen_q,
#                             max_seqlen_k=max_seqlen_k,
#                             dropout_p=0.0,
#                             causal=True,
#                         ).to(self.dtype)
#                 else:
#                     output = flash_attn_varlen_kvpacked_func(
#                         q=total_q,
#                         kv=total_kv,
#                         cu_seqlens_q=cu_seqlens,
#                         cu_seqlens_k=cu_seqlens,
#                         max_seqlen_q=max_seqlen_q,
#                         max_seqlen_k=max_seqlen_k,
#                         dropout_p=0.0,
#                         causal=True,
#                     )

#                 context = torch.zeros_like(q)
#                 context = context.masked_scatter_(attn_mask4flsh.view(bsz, -1, 1, 1), output)

#             else:
#                 attn_mask = inference_params.attention_mask[:, -1, :].view(bsz, 1, 1, -1)
#                 if hasattr(inference_params, "window_size") and inference_params.window_size is not None:
#                     if inference_params.window_size <= inference_params.sequence_len_offset:
#                         attn_mask = torch.concat(
#                             [
#                                 attn_mask[..., : inference_params.keep_first],
#                                 attn_mask[..., -(inference_params.window_size - inference_params.keep_first) :],
#                             ],
#                             dim=-1,
#                         )

#                 k, v = torch.chunk(kv, 2, dim=2)
#                 k = k.squeeze(2)
#                 v = v.squeeze(2)
#                 sp = k.shape
#                 expansion = q.size(2) // k.size(2)
#                 scores = torch.einsum(
#                     "blhd,bnhd->bhln",
#                     q,
#                     k.unsqueeze(3).expand(-1, -1, -1, expansion, -1).reshape(sp[0], sp[1], q.size(2), sp[3]),
#                 ) / math.sqrt(q.size(-1))
#                 scores = scores.masked_fill(attn_mask, -65000.0)
#                 scores = F.softmax(scores, dim=-1)  # bsz x h x L x L
#                 context = torch.einsum(
#                     "bhmn,bnhd->bmhd",
#                     scores,
#                     v.unsqueeze(3).expand(-1, -1, -1, expansion, -1).reshape(sp[0], sp[1], q.size(2), sp[3]),
#                 )
#         else:
#             if self.dtype is torch.float32 and self.use_flash_attn:
#                 if q.dtype not in [torch.float16, torch.bfloat16]:
#                     q = q.to(torch.bfloat16)
#                 if kv.dtype not in [torch.float16, torch.bfloat16]:
#                     kv = kv.to(torch.bfloat16)
#                 with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#                     context = self.inner_cross_attn(q, kv, causal=True).to(self.dtype)
#             else:
#                 context = self.inner_cross_attn(q, kv, causal=True)


class QKVPackedMHA(nn.Module):
    """
    Multi-head self-attention and cross-attention.

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        process_group (torch.distributed.ProcessGroup): The group of the current device for `parallel_mode`.
        max_position_embeddings (int): max position embeddings, 2048 by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        softmax_scale (float): The temperature to use for the softmax attention.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        use_dynamic_ntk_rope (bool): whether use dynamic ntk rope, false by default.
        rotary_emb_dim (int): The dimention of Rotary Embedding. 0 by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        use_flash_attn (bool): Whether to use flash-attn. True by default.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_position_embeddings: int = 2048,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        use_dynamic_ntk_rope: bool = False,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        rope_base: int = 10000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.causal = causal
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope
        self.rotary_emb_dim = rotary_emb_dim
        self.max_position_embeddings = max_position_embeddings
        factory_kwargs = {"device": device, "dtype": dtype}

        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"

        if self.rotary_emb_dim > 0:
            self.rotary_emb = new_rotary_embedding(
                self.rotary_emb_dim,
                base=rope_base,
                scale_base=rotary_emb_scale_base,
                device=device,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=1.0,
                rotary_type="dynamic_ntk" if self.use_dynamic_ntk_rope else "native",
            )

        # bias=True is according to https://spaces.ac.cn/archives/9577
        self.Wqkv = new_linear(
            "Wqkv", embed_dim, 3 * embed_dim, bias=True, **factory_kwargs
        )

        self.inner_attn = SelfAttention(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.inner_cross_attn = CrossAttention(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )

        # output projection always have the bias (for now)
        self.out_proj = new_linear(
            "out_proj", embed_dim, embed_dim, bias=True, **factory_kwargs
        )

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        if inference_params is not None:
            return self._inference(x=x, inference_params=inference_params, **kwargs)
        else:
            return self._training(
                x=x, seqlen=seqlen, inference_params=inference_params, **kwargs
            )

    def _inference(self, x, inference_params, **kwargs):
        assert False, "NYI"

    def _training(self, x, **kwargs):  # pylint: disable=W0613
        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )

        q = qkv[..., 0, :, :].squeeze(2)
        k = qkv[..., 1, :, :].squeeze(2)

        indexes = kwargs.pop("indexes", 0)
        self.rotary_emb(q, offsets=indexes, cache_type="query", in_place=True)
        self.rotary_emb(k, offsets=indexes, cache_type="key", in_place=True)
        # if gpc.config.model.dtype is torch.float32 and gpc.config.model.use_flash_attn:
        #     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        #         if qkv.dtype not in [torch.float16, torch.bfloat16]:
        #             qkv = qkv.to(torch.bfloat16)
        #         context = self.inner_attn(qkv).to(x.dtype)
        # else:
        context = self.inner_attn(qkv)

        # else:
        #     # TODO
        #     pass

        # 我们似乎没有需求对seqlen进行处理，根据注释他与sequence parallel有关
        # sq也不应该在这里处理
        # if seqlen is None:
        # context = rearrange(context, "b s h d -> b s (h d)")
        # else:
        # context = rearrange(context, "b s h d -> (b s) (h d)")
        context = rearrange(context, "... h d -> ... (h d)")

        out = self.out_proj(context)
        return out


# qkv分离，GQA
class QKVSplitedGQA(nn.Module):
    """
    Multi-head self-attention and cross-attention.

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        process_group (torch.distributed.ProcessGroup): The group of the current device for `parallel_mode`.
        sequence_process_group (torch.distributed.ProcessGroup): The process group for attention calculation.
        bias (boolean): Whether the bias is needed for linears. Will be used when initializing QKV matrix and
                        output projection. True by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        softmax_scale (float): The temperature to use for the softmax attention.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        rotary_emb_dim (int): The dimention of Rotary Embedding. 0 by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        use_flash_attn (boolean): Whether to use flash attention or not.If False, vanilla attention module will be used.
                                    False by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        use_flash_attn (bool): Whether to use flash-attn. True by default.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        tp_mode (str): The string value of tensor parallel mode, should be in ["mtp", "msp", "fsp", "isp"],
                       "mtp" by default.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        # process_group: Optional[torch.distributed.ProcessGroup],
        # sequence_process_group: Optional[torch.distributed.ProcessGroup],
        bias: bool = True,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        rope_base: int = 10000,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        # use_flash_attn: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        rot_embed_HF_impl: Optional[bool] = False,
        # tp_mode: str = "mtp",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.causal = causal
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = self.embed_dim // num_heads
        self.kv_dim = self.head_dim * num_kv_heads
        self.rotary_emb_dim = rotary_emb_dim
        # self.use_flash_attn = use_flash_attn
        # self.dtype = dtype
        # self.tp_mode = tp_mode
        self.rot_embed_HF_impl = rot_embed_HF_impl
        factory_kwargs = {"device": device, "dtype": dtype}

        assert (
            self.embed_dim % num_heads == 0
        ), "embedding dim must be divisible by num_heads"

        if self.rotary_emb_dim > 0:
            self.rotary_emb = new_rotary_embedding(
                self.rotary_emb_dim,
                base=rope_base,
                scale_base=rotary_emb_scale_base,
                device=device,
                rotary_type="native",
            )

        # notice here should change bias=True
        self.wq = new_linear("wq", embed_dim, embed_dim, bias, **factory_kwargs)
        self.wk = new_linear("wk", embed_dim, self.kv_dim, bias, **factory_kwargs)
        self.wv = new_linear("wv", embed_dim, self.kv_dim, bias, **factory_kwargs)

        # if use_flash_attn:
        #     from flash_attn import flash_attn_varlen_kvpacked_func
        #     from flash_attn.modules.mha import FlashCrossAttention, FlashSelfAttention

        # inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        # inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        self.inner_attn = SelfAttention(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.inner_cross_attn = CrossAttention(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )

        # self.inner_cross_attn_causal = causal
        # self.inner_cross_attn_softmax_scale = softmax_scale
        # self.inner_cross_attn_dropout = dropout

        # self.attn = flash_attn_varlen_kvpacked_func if use_flash_attn else SelfAttention
        # if self.tp_mode == "isp":
        #     self.attn = DistributedAttention(self.attn, sequence_process_group=sequence_process_group)

        # output projection always have the bias (for now)
        self.wo = new_linear("wo", embed_dim, embed_dim, bias, **factory_kwargs)

    def forward(self, x, inference_params=None, **kwargs):
        if inference_params is not None:
            return self._inference(x=x, inference_params=inference_params, **kwargs)
        else:
            return self._training(x=x, **kwargs)

    # def forward(self, x, seqlen=None, inference_params=None, **kwargs):
    #     if kwargs.get("indexes", None) is not None:
    #         return self._packed_forward(x=x, inference_params=inference_params, **kwargs)
    #     else:
    #         return self._forward(x=x, seqlen=seqlen, inference_params=inference_params, **kwargs)

    def _inference(self, x, inference_params, **kwargs):
        assert False, "NYI"

    def _training(self, x, **kwargs):  # pylint: disable=W0613
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        if self.rotary_emb_dim > 0:
            indexes = kwargs.pop("indexes", 0)

            if not self.rot_embed_HF_impl:
                q = torch.cat([q[..., ::2], q[..., 1::2]], dim=-1)
                k = torch.cat([k[..., ::2], k[..., 1::2]], dim=-1)

            q = self.rotary_emb(q, offsets=indexes, cache_type="query")
            k = self.rotary_emb(k, offsets=indexes, cache_type="key")

        kv = torch.concat(
            [k.unsqueeze(2), v.unsqueeze(2)], dim=2
        )  # data-pack data-unpack统一

        context = self.inner_attn(q, kv, **kwargs)
        context = rearrange(context, "b s h d -> b s (h d)")

        out = self.wo(context)
        return out


# qkv合并，GQA
class QKVPackedGQA(nn.Module):
    """
    Multi-head self-attention and cross-attention.

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        num_kv_heads (int): The number of attention heads for key and value.
        process_group (torch.distributed.ProcessGroup): The group of the current device for `parallel_mode`.
        sequence_process_group (torch.distributed.ProcessGroup): The process group for attention calculation.
        bias (bool): Whether the bias is needed for linears. Will be used when initializing QKV matrix and
                     output projection. False by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        softmax_scale (float): The temperature to use for the softmax attention.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        rotary_emb_dim (int): The dimention of Rotary Embedding. 0 by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        use_flash_attn (bool): Whether to use flash attention or not.If False, vanilla attention module will be used.
                               False by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        rot_embed_HF_impl (Optional[bool]): Whether to use the rotary embedding implementation from HuggingFace.
                                            True by default.
        tp_mode (str): The string value of tensor parallel mode, should be in ["mtp", "msp", "fsp", "isp"],
                       "mtp" by default.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        # process_group: Optional[torch.distributed.ProcessGroup],
        # sequence_process_group: Optional[torch.distributed.ProcessGroup],
        max_position_embeddings: int = 2048,
        bias: bool = False,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        use_dynamic_ntk_rope: bool = False,
        # use_flash_attn: bool = True,
        rope_base: int = 10000,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        rot_embed_HF_impl: Optional[bool] = True,
        # tp_mode: str = "mtp",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.causal = causal

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_per_kv = num_heads // num_kv_heads
        self.head_dim = self.embed_dim // num_heads
        self.kv_dim = self.head_dim * num_kv_heads
        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope
        self.rotary_emb_dim = rotary_emb_dim
        self.max_position_embeddings = max_position_embeddings
        self.rot_embed_HF_impl = rot_embed_HF_impl
        factory_kwargs = {"device": device, "dtype": dtype}

        assert (
            self.embed_dim % num_heads == 0
        ), "embedding dim must be divisible by num_heads"

        if self.rotary_emb_dim > 0:
            self.rotary_emb = new_rotary_embedding(
                self.rotary_emb_dim,
                base=rope_base,
                scale_base=rotary_emb_scale_base,
                device=device,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=1.0,
                rotary_type="dynamic_ntk" if self.use_dynamic_ntk_rope else "native",
            )

        self.wqkv = new_linear(
            "wqkv", embed_dim, embed_dim + 2 * self.kv_dim, bias, **factory_kwargs
        )

        self.inner_attn = SelfAttention(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.inner_cross_attn = CrossAttention(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )

        self.wo = new_linear("wo", embed_dim, embed_dim, bias, **factory_kwargs)

    def forward(self, x, inference_params=None, **kwargs):
        if inference_params is not None:
            return self._inference(x=x, inference_params=inference_params, **kwargs)
        else:
            return self._training(x=x, **kwargs)

    # def forward(self, x, seqlen=None, inference_params=None, **kwargs):
    #     if kwargs.get("indexes", None) is not None:
    #         return self._packed_forward(
    #             x=x, inference_params=inference_params, **kwargs
    #         )
    #     else:
    #         return self._forward(
    #             x=x, seqlen=seqlen, inference_params=inference_params, **kwargs
    #         )

    def _inference(self, x, inference_params, **kwargs):
        assert False, "NYI"

    def _training(self, x, **kwargs):  # pylint: disable=W0613
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        qkv = self.wqkv(x)

        qkv = rearrange(
            qkv, "b s (h gs d) -> b s h gs d", gs=self.q_per_kv + 2, d=self.head_dim
        )

        q, k, v = (qkv[..., : self.q_per_kv, :], qkv[..., -2, :], qkv[..., -1, :])

        q = rearrange(q, "b s h gs d -> b s (h gs) d")

        if self.rotary_emb_dim > 0:
            indexes = kwargs.pop("indexes", 0)

            if not self.rot_embed_HF_impl:
                q = torch.cat([q[..., ::2], q[..., 1::2]], dim=-1)
                k = torch.cat([k[..., ::2], k[..., 1::2]], dim=-1)

            q = self.rotary_emb(q, offsets=indexes, cache_type="query")
            k = self.rotary_emb(k, offsets=indexes, cache_type="key")

        kv = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)

        context = self.inner_attn(q, kv, **kwargs)

        context = rearrange(context, "b s h d -> b s (h d)")

        out = self.wo(context)

        return out
