"""

Inspired by,
* https://github.com/karpathy/nanoGPT/tree/master
* https://theaisummer.com/einsum-attention/
* https://arxiv.org/abs/2207.09238
* https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
* https://github.com/Dao-AILab/flash-attention/tree/main


Notes on padding, packing, and masking
* https://github.com/pytorch/pytorch/issues/103749
* https://github.com/Dao-AILab/flash-attention/issues/432#issuecomment-1668822286


"""
from enum import Enum
import math
from typing import Optional

import einops
from einops import rearrange
from flash_attn import flash_attn_qkvpacked_func
from flash_attn import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input
from flash_attn.bert_padding import pad_input
from pydantic import BaseModel
from pydantic import Field
import rich
import torch
import torch.nn as nn
from torch.nn import functional as F


class FormalLamoConfig(BaseModel):
    n_vocab: int = Field(gt=0, description="number of tokens in vocabulary")
    padding_idx: Optional[int] = Field(description="index of padding token")
    n_layer: int = Field(gt=0, default=12, description="number of layers")
    n_head: int = Field(gt=0, default=12, description="number of attention heads")
    l_max: int = Field(gt=0, default=512, description="maximum sequence length")
    d_x: int = Field(gt=0, default=768, description="embedding size for primary sequence")
    d_z: int = Field(gt=0, default=768, description="embedding size for context sequence")
    d_attn: int = Field(gt=0, default=64, description="embedding size per head for query and key projections")
    d_mid: int = Field(gt=0, default=64, description="embedding size per head for value projection")
    d_out: int = Field(gt=0, default=768, description="embedding size for output")
    fc_mult: int = Field(gt=0, default=4, description="fully connected layer multiplier")
    pre_layernorm: bool = Field(default=True, description="pre layernorm if true else post layernorm")
    dropout: float = Field(ge=0, default=0.0, description="dropout probability")
    bias: bool = Field(default=False, description="use bias parameters if true")
    pos_type: PosType = Field(default="learned", description="type of position embeddings")
    tie_weights: bool = Field(default=True, description="tie input embedding and lm head weights")
    attn_impl: AttnImpl = Field(default="flash_varlen_qkvpacked", description="attention implementation")


class FormalMultiheadAttention(nn.Module):

    def __init__(self, config: FormalLamoConfig):

        super().__init__()
        self.config = config
        self.w_q = nn.Linear(config.d_x, config.n_head * config.d_attn, bias=config.bias)
        self.w_k = nn.Linear(config.d_z, config.n_head * config.d_attn, bias=config.bias)
        self.w_v = nn.Linear(config.d_z, config.n_head * config.d_mid, bias=config.bias)
        self.w_o = nn.Linear(config.n_head * config.d_mid, config.d_out, bias=config.bias)
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, x, z, attn_mask=None, is_causal=False):

        bs, lx, dx = x.shape
        bsz, lz, dz = z.shape

        assert bs == bsz
        assert dx == self.config.d_x
        assert dz == self.config.d_z
        if attn_mask is not None:
            amask_check_1 = attn_mask.shape == (bs, 1, lx, lz)
            amask_check_2 = attn_mask.shape == (bs, self.config.n_head, lx, lz)
            assert amask_check_1 or amask_check_2

        qq = rearrange(self.w_q(x), "bs lx (nh da) -> bs nh lx da", nh=self.config.n_head)
        kk = rearrange(self.w_k(z), "bs lz (nh da) -> bs nh lz da", nh=self.config.n_head)
        vv = rearrange(self.w_v(z), "bs lz (nh dm) -> bs nh lz dm", nh=self.config.n_head)

        dropout_p = self.config.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            qq, kk, vv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )
        y = rearrange(y, "bs nh lx dm -> bs lx (nh dm)")
        y = self.output_dropout(self.w_o(y))

        return y


