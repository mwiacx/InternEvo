#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from typing import Optional

import torch
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.core.naive_amp import set_output_attr_to_module
from internlm.initialize.initialize_tensor import normal_, scaled_init_method_normal
from internlm.model.modules.embedding import Embedding1D
from internlm.model.modules.linear import new_linear
from internlm.model.modules.mha import MHA
from internlm.model.modules.mlp import new_feed_forward
from internlm.model.modules.norm import new_layer_norm
from internlm.model.utils import (
    internlm1_mha_pre_load_convert,
    internlm1_mha_save_convert,
)
from internlm.solver.activation_checkpoint import activation_checkpoint
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


class InternLM1Decoder(nn.Module):
    """
    1D Packed Flash Base Layer.

    Args:
        hidden_size (int): The hidden size of model. 768 by default.
        num_attention_heads (int): The number of attention heads. 12 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0 by default.
        drop_rate (float): The dropout rate of the input hidden state. 0.0 by default.
        max_position_embeddings (int): The maximum position embeddings. 2048 by default.
        dtype (torch.dtype): Type of data. torch.float by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        layer_idx (int): The index of current layer. 0 by default.
        use_dynamic_ntk_rope (bool): Whether to use dynamic ntk rope. False by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        norm_type (str): Use RMS norm or layernorm."rmsnorm" by default.
        qk_interleaved (bool): Whether the odd and even columns of the wq and wk are normally interleaved.
        dropout_selective_checkpoint (bool): Whether to selectively checkpoint dropout layers only.
        use_scaled_init (bool): Whether to use scaled initialization for weights.
        use_swiglu (bool): Whether to use SwiGLU activation in the mlp module.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        mlp_layer_fusion (bool): Whether to fuse layers in the mlp module for optimization.
        multiple_of (int): Ensures mlp dimensions are multiples of this value for efficient hardware utilization.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0,
        drop_rate: float = 0.0,
        max_position_embeddings: int = 2048,
        dtype: torch.dtype = torch.float,
        layer_norm_epsilon: float = 1e-6,
        checkpoint: bool = False,
        layer_idx: int = 0,
        use_dynamic_ntk_rope: bool = False,
        residual_in_fp32: bool = False,
        device: Optional[torch.device] = None,
        norm_type: str = "rmsnorm",
        qk_interleaved: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        rope_base: int = 10000,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        # dropout selective checkpoint can only be enabled when checkpoint is disabled.
        self.dropout_selective_checkpoint = dropout_selective_checkpoint is True and checkpoint is False
        self.layer_idx = layer_idx

        head_dim = hidden_size // num_attention_heads

        self.mixer = MHA(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attn_drop_rate,
            max_position_embeddings=max_position_embeddings,
            softmax_scale=1 / math.sqrt(head_dim),
            causal=True,
            layer_idx=layer_idx,
            use_dynamic_ntk_rope=use_dynamic_ntk_rope,
            rotary_emb_dim=head_dim,
            rotary_emb_scale_base=0,
            rope_base=rope_base,
            device=device,
            dtype=dtype,
            qk_interleaved=qk_interleaved,
        )

        # Compatible with the name of internlm1 Wqkv linear layer
        self.mixer.register_checkpoint_compatibility_hooks(internlm1_mha_pre_load_convert, internlm1_mha_save_convert)

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)

        self.norm1 = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)
        self.norm2 = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)

        self.mlp = new_feed_forward(
            hidden_size,
            int(hidden_size * mlp_ratio),
            out_features=hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
            mlp_layer_fusion=mlp_layer_fusion,
            multiple_of=multiple_of,
            # TODO: to support more activation functions
            activation_type="swiglu" if use_swiglu else "swiglu",
        )

        self.use_swiglu = use_swiglu
        self.use_scaled_init = use_scaled_init
        self.residual_in_fp32 = residual_in_fp32  # only make sense when using prenorm
        self.return_residual = False
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.mixer.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "wqkv" in name:
                    normal_(std=0.006)(param.data)
                elif self.use_scaled_init:
                    scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                else:
                    normal_(std=0.0015)(param.data)

            for name, param in self.mlp.named_parameters():
                if param.ndim == 1 and "bias" in name:
                    param.data.zero_()
                elif self.use_swiglu:
                    if self.use_scaled_init and "w2" in name:
                        scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        # candidate: w1, w3, fused_w1_w3
                        normal_(std=0.006 if "w1" in name or "w3" in name else 0.0015)(param.data)
                else:
                    if self.use_scaled_init and "fc1" not in name:
                        scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        normal_(std=0.006 if "fc1" in name else 0.0015)(param.data)

    def forward(self, hidden_states, **kwargs):
        if self.checkpoint and self.training:
            return activation_checkpoint(self._forward, False, hidden_states, **kwargs)
        else:
            return self._forward(hidden_states, **kwargs)

    def _forward(self, hidden_states=None, **kwargs):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Attn/MLP(LN(residual))
            cu_seqlens: 1d LongTensor, len(cu_seqlens) = hidden_states + 1
            indexes: the length of index is same as hidden states, which stand for the current position
        """

        def _dropout_and_norm_attn(_hidden_states):
            _dropped = self.dropout1(_hidden_states)
            _residual = _dropped
            _hidden_states = self.norm1(_residual.float())
            return _residual, _hidden_states

        if self.dropout_selective_checkpoint:
            residual, hidden_states = activation_checkpoint(_dropout_and_norm_attn, False, hidden_states)
        else:
            residual, hidden_states = _dropout_and_norm_attn(hidden_states)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, **kwargs)

        def _dropout_and_norm_ffn(_residual, _hidden_states):
            _dropped = self.dropout2(_hidden_states)
            _residual = (_dropped + _residual) if _residual is not None else _dropped
            _hidden_states = self.norm2(_residual.float())
            return _residual, _hidden_states

        if self.dropout_selective_checkpoint:
            residual, hidden_states = activation_checkpoint(_dropout_and_norm_ffn, False, residual, hidden_states)
        else:
            residual, hidden_states = _dropout_and_norm_ffn(residual, hidden_states)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mlp(hidden_states)

        return hidden_states + residual


class InternLM1(nn.Module):
    """
    1D Packed Flash InternLm.

    Args:
        num_layers (int): The number of layer. 12 by default.
        hidden_size (int): The size of hidden state. 768 by default.
        num_attention_heads (int): The number of attention head. 12 by default.
        vocab_size (int): The size of vocabulary. 50304 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0.0 by default.
        drop_rate (float): The dropout rate of input hidden state. 0.0 by default.
        max_position_embeddings (int): The maximum position embeddings. 2048 by default.
        dtype (torch.dtype): The type of data. torch.float by default.
        checkpoint (float): The proportion of layers that need to be checkpointed compared to the total number
                                    of layers. 0.0 by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        first (bool): Whether input embedding layer or not. False by default.
        last (bool): Whether output embedding layer or not. False by default.
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default.
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default.
        start_layer_idx (int): The index of start layer in the pipeline. 0 by default.
        use_dynamic_ntk_rope (bool): Whether to use dynamic ntk rope. False by default.
        device (Optional[Union[str, torch.device]]): The device will be used. None by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default.
        qk_interleaved (bool): Whether the odd and even columns of the wq and wk are normally interleaved.
        dropout_selective_checkpoint (bool): Whether to selectively checkpoint dropout and norm layers.
        use_scaled_init (bool): Whether to use scaled initialization for weights.
        use_swiglu (bool): Whether to use SwiGLU activation in the mlp module.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        mlp_layer_fusion (bool): Whether to fuse layers in the mlp module for optimization.
        multiple_of (int): Ensures mlp dimensions are multiples of this value for efficient hardware utilization.
    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        vocab_size: int = 50304,
        mlp_ratio: int = 4.0,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        max_position_embeddings: int = 2048,
        dtype: torch.dtype = torch.float,
        checkpoint: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        first: bool = False,
        last: bool = False,
        embed_grad_scale: float = 0.1,
        parallel_output: bool = True,
        start_layer_idx: int = 0,
        use_dynamic_ntk_rope: bool = False,
        device: Optional[torch.device] = None,
        residual_in_fp32: bool = False,
        norm_type: str = "rmsnorm",
        qk_interleaved: bool = False,
        is_reward: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        rope_base: int = 10000,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
    ):
        super().__init__()

        checkpoint_layer_num = int(num_layers * checkpoint)
        self.embed_grad_scale = embed_grad_scale
        self.parallel_output = parallel_output

        if first:
            self.embedding = Embedding1D(num_embeddings=vocab_size, embedding_dim=hidden_size)
            for _, param in self.embedding.named_parameters():
                normal_(std=0.0052)(param)

        self.blocks = nn.ModuleList(
            [
                InternLM1Decoder(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    max_position_embeddings=max_position_embeddings,
                    dtype=dtype,
                    layer_norm_epsilon=layer_norm_epsilon,
                    checkpoint=lid < checkpoint_layer_num,
                    layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
                    use_dynamic_ntk_rope=use_dynamic_ntk_rope,
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    norm_type=norm_type,
                    dropout_selective_checkpoint=dropout_selective_checkpoint,
                    use_scaled_init=use_scaled_init,
                    use_swiglu=use_swiglu,
                    rope_base=rope_base,
                    qk_interleaved=qk_interleaved,
                    mlp_layer_fusion=mlp_layer_fusion,
                    multiple_of=multiple_of,
                )
                for lid in range(num_layers)
            ]
        )

        if last:
            self.norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)
            self.head = new_linear(
                name="head",
                in_features=hidden_size,
                out_features=gpc.get_world_size(ParallelMode.TENSOR) if is_reward else vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
                is_reward=is_reward,
                weight_scale=embed_grad_scale,
            )
            set_output_attr_to_module(self.head)
            for _, param in self.head.named_parameters():
                normal_(std=0.0052)(param)

    def forward(self, hidden_states=None, input_ids=None, **kwargs):
        # attention_mask: compute attention on the places where the value is 1
        if hasattr(self, "embedding") and input_ids is not None:
            hidden_states = self.embedding(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )

        for _, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, **kwargs)

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states.float())
        if hasattr(self, "head"):
            hidden_states = self.head(hidden_states)

        return hidden_states
