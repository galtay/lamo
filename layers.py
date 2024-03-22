"""
Inspired by,
* https://github.com/karpathy/nanoGPT/tree/master
* https://theaisummer.com/einsum-attention/
* https://arxiv.org/abs/2207.09238
* https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch

Notes on padding and masking
* https://github.com/pytorch/pytorch/issues/103749


TODO: tie embedding weights

"""
from enum import Enum
import math
from typing import Optional

import einops
from pydantic import BaseModel
from pydantic import Field
import rich
import torch
import torch.nn as nn
from torch.nn import functional as F


class PosEmbType(str, Enum):
    none = "none"
    learned = "learned"


class LamoConfig(BaseModel):
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
    pos_emb_type: PosEmbType = Field(default="learned", description="type of position embeddings")


class MultiheadAttention(nn.Module):

    def __init__(self, config: LamoConfig):

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

        qq = einops.rearrange(self.w_q(x), "bs lx (nh da) -> bs nh lx da", nh=self.config.n_head)
        kk = einops.rearrange(self.w_k(z), "bs lz (nh da) -> bs nh lz da", nh=self.config.n_head)
        vv = einops.rearrange(self.w_v(z), "bs lz (nh dm) -> bs nh lz dm", nh=self.config.n_head)

        dropout_p = self.config.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            qq, kk, vv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )
        y = einops.rearrange(y, "bs nh lx dm -> bs lx (nh dm)")
        y = self.output_dropout(self.w_o(y))

        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.w_fc_in = nn.Linear(config.d_x, config.fc_mult * config.d_x, bias=config.bias)
        self.gelu = nn.GELU()
        self.w_fc_out = nn.Linear(config.fc_mult * config.d_x, config.d_x, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.w_fc_in(x)
        x = self.gelu(x)
        x = self.w_fc_out(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.d_x, bias=config.bias)
        self.attn = MultiheadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_x, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None, is_causal=False):
        if self.config.pre_layernorm:
            ln1 = self.ln_1(x)
            x = x + self.attn(ln1, ln1, attn_mask=attn_mask, is_causal=is_causal)
            x = x + self.mlp(self.ln_2(x))
        else:
            x = self.ln_1(x + self.attn(x, x, attn_mask))
            x = self.ln_2(x + self.mlp(x))
        return x


class LamoEncoder(nn.Module):

    def __init__(self, config: LamoConfig):

        super().__init__()
        self.config = config
        assert config.d_x == config.d_z

        wte = nn.Embedding(config.n_vocab, config.d_x, padding_idx=config.padding_idx)

        if config.pos_emb_type == "none":
            self.encoder = nn.ModuleDict(
                {
                    "wte": wte,
                    "dropout": nn.Dropout(config.dropout),
                    "blocks": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                    "ln_f": nn.LayerNorm(config.d_x, bias=config.bias),
                }
            )
        elif config.pos_emb_type == "learned":
            wpe = nn.Embedding(config.l_max, config.d_x)
            self.encoder = nn.ModuleDict(
                {
                    "wte": wte,
                    "wpe": wpe,
                    "dropout": nn.Dropout(config.dropout),
                    "blocks": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                    "ln_f": nn.LayerNorm(config.d_x, bias=config.bias),
                }
            )

        self.lm_head = nn.Linear(config.d_x, config.n_vocab, bias=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("w_o.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))


    def forward(self, input_ids, attention_mask):

        bs, l_x = input_ids.shape
        assert attention_mask.shape == (bs, l_x)
        assert l_x <= self.config.l_max

        # convert to [bs, l_x, l_x] by repeating each sequence l_x times
        # then make boolean and expand so it is broadcastable to [bs, nh, l_x, l_x]
        attn_mask = attention_mask.repeat(1, 1, l_x).reshape(bs, l_x, l_x)
        attn_mask = attn_mask == 1
        attn_mask = attn_mask[:, None, :, :]

        tok_emb = self.encoder.wte(input_ids)
        if self.config.pos_emb_type == "none":
            emb = tok_emb
        elif self.config.pos_emb_type == "learned":
            pos = torch.arange(0, l_x, dtype=torch.long, device=input_ids.device)
            pos_emb = self.encoder.wpe(pos)
            emb = tok_emb + pos_emb

        x = self.encoder.dropout(emb)
        for block in self.encoder.blocks:
            x = block(x, attn_mask)
        x = self.encoder.ln_f(x)
        logits = self.lm_head(x)

        return {
            "attn_mask": attn_mask,
            "logits": logits,
        }

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


if __name__ == "__main__":

    d_e = 768
    n_head = 12
    d_mid = d_e // n_head
    d_attn = d_e // n_head
    fc_mult = 4
    n_vocab = 2**15
    pre_layernorm = True

    config_encoder = LamoConfig(
        n_layer=12,
        n_vocab=n_vocab,
        n_head=n_head,
        l_max=512,
        d_x=d_e,
        d_z=d_e,
        d_attn=d_attn,
        d_mid=d_mid,
        d_out=d_e,
        fc_mult=fc_mult,
        pre_layernorm=pre_layernorm,
        padding_idx=0,
        dropout=0.0,
        bias=False,
    )

    config = LamoConfig(
        n_layer=12,
        n_vocab=n_vocab,
        n_head=n_head,
        l_max=512,
        d_x=n_head * 2,
        d_z=n_head * 3,
        d_attn=n_head * 4,
        d_mid=n_head * 5,
        d_out=n_head * 6,
        fc_mult=fc_mult,
        pre_layernorm=pre_layernorm,
        padding_idx=0,
        dropout=0.0,
        bias=False,
    )

    mha = MultiheadAttention(config)

    b_s = 3
    l_x = config.l_max
    l_z = config.l_max
    x = torch.rand(b_s, l_x, config.d_x)
    z = torch.rand(b_s, l_z, config.d_z)
    mha_out = mha(x, z)

    input_ids = torch.randint(0, n_vocab, size=(b_s, l_x))
    attention_mask = torch.ones_like(input_ids)
    lamo = LamoEncoder(config_encoder)
    rich.print("number of parameters: %.4fM" % (lamo.get_num_params() / 1e6,))
    lamo_out = lamo(input_ids, attention_mask)
