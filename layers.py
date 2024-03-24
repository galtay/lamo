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


def get_alibi_slopes(nheads: int):

    def get_slopes_power_of_2(nheads: int):
        start = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        ratio = start
        return [start * ratio**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][: nheads - closest_power_of_2]
        )


class PosType(str, Enum):
    none = "none"
    learned = "learned"
    alibi = "alibi"

class AttnImpl(str, Enum):
    torch_sdpa = "torch_spda"
    flash_varlen_qkvpacked = "flash_varlen_qkvpacked"


class LamoConfig(BaseModel):
    n_vocab: int = Field(gt=0, description="number of tokens in vocabulary")
    padding_idx: Optional[int] = Field(description="index of padding token")
    n_layer: int = Field(gt=0, default=12, description="number of layers")
    n_head: int = Field(gt=0, default=12, description="number of attention heads")
    l_max: int = Field(gt=0, default=512, description="maximum sequence length")
    d_e: int = Field(gt=0, default=768, description="embedding size")
    fc_mult: int = Field(gt=0, default=4, description="fully connected layer multiplier")
    pre_layernorm: bool = Field(default=True, description="pre layernorm if true else post layernorm")
    dropout: float = Field(ge=0, default=0.0, description="dropout probability")
    bias: bool = Field(default=False, description="use bias parameters if true")
    pos_type: PosType = Field(default="alibi", description="how to handle position information")
    tie_weights: bool = Field(default=True, description="tie input embedding and lm head weights")
    attn_impl: AttnImpl = Field(default="flash_varlen_qkvpacked", description="attention implementation")

class MultiheadAttention(nn.Module):

    def __init__(self, config: LamoConfig):

        super().__init__()
        assert (
            config.d_e % config.n_head == 0
        ), "d_e is not divisible by n_head"
        self.config = config
        self.d_head = config.d_e // config.n_head
        self.w_qkv = nn.Linear(config.d_e, 3 * config.d_e, bias=config.bias)
        self.w_o = nn.Linear(config.d_e, config.d_e, bias=config.bias)
        self.output_dropout = nn.Dropout(config.dropout)
        if config.pos_type == "alibi":
            self.alibi_slopes = torch.tensor(get_alibi_slopes(config.n_head))
        else:
            self.alibi_slopes = None

    def forward_flash_attn_varlen_qkvpacked_func(self, x, attn_mask=None, is_causal=False):
        bs, lx, de = x.shape
        n3 = 3
        nh = self.config.n_head
        dh = de // nh
        assert de == self.config.d_e
        assert attn_mask.shape == (bs, lx)

        qkv = self.w_qkv(x)
        assert qkv.shape == (bs, lx, n3 * de)

        hidden_states = rearrange(qkv, "bs lx (n3 nh dh) -> bs lx n3 nh dh", n3=n3, nh=nh)
        hidden_states_u, indices, cu_seqlens, max_seqlen = unpad_input(hidden_states, attn_mask)

        assert attn_mask.sum() == len(indices)
        n_tok = len(indices)
        assert hidden_states_u.shape == (n_tok, n3, nh, dh)

        dropout_p = self.config.dropout if self.training else 0.0
        y_u = flash_attn_varlen_qkvpacked_func(
            hidden_states_u,
            cu_seqlens,
            max_seqlen,
            dropout_p = dropout_p,
            causal=is_causal,
            alibi_slopes = None if self.alibi_slopes is None else self.alibi_slopes.to(x.device),
        )
        assert y_u.shape == (n_tok, nh, dh)

        y = pad_input(y_u, indices, bs, max_seqlen)
        assert y.shape == (bs, lx, nh, dh)

        y = rearrange(y, "bs lx nh dh -> bs lx (nh dh)")
        assert y.shape == (bs, lx, de)

        y = self.output_dropout(self.w_o(y))
        return y

    def forward_scaled_dot_product_attention(self, x, attn_mask=None, is_causal=False):
        bs, lx, de = x.shape
        n3 = 3
        nh = self.config.n_head
        dh = de // nh
        assert de == self.config.d_e
        assert attn_mask.shape == (bs, lx)

        qkv = self.w_qkv(x)
        assert qkv.shape == (bs, lx, n3 * de)

        if attn_mask is not None:

            # convert to [bs, l_x, l_x] by repeating each sequence l_x times
            # then make boolean and expand so it is broadcastable to [bs, nh, l_x, l_x]
            amask = attn_mask.repeat(1, 1, lx).reshape(bs, lx, lx)
            amask = amask == 1
            amask = amask[:, None, :, :]

            amask_check_1 = amask.shape == (bs, 1, lx, lx)
            amask_check_2 = amask.shape == (bs, nh, lx, lx)
            assert amask_check_1 or amask_check_2

        qq, kk, vv = tuple(rearrange(qkv, "bs lx (n3 nh dh) -> n3 bs nh lx dh", n3=n3, nh=nh))
        dropout_p = self.config.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            qq, kk, vv, attn_mask=amask, dropout_p=dropout_p, is_causal=is_causal
        )
        y = rearrange(y, "bs nh lx dh -> bs lx (nh dh)")

        y = self.output_dropout(self.w_o(y))
        return y

    def forward(self, x, attn_mask=None, is_causal=False):

        if self.config.attn_impl == "flash_varlen_qkvpacked":
            y = self.forward_flash_attn_varlen_qkvpacked_func(x, attn_mask=attn_mask, is_causal=is_causal)
        elif self.config.attn_impl == "torch_spda":
            y = self.forward_scaled_dot_product_attention(x, attn_mask=attn_mask, is_causal=is_causal)
        else:
            raise ValueError()
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.w_fc_in = nn.Linear(config.d_e, config.fc_mult * config.d_e, bias=config.bias)
        self.gelu = nn.GELU()
        self.w_fc_out = nn.Linear(config.fc_mult * config.d_e, config.d_e, bias=config.bias)
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
        self.ln_1 = nn.LayerNorm(config.d_e, bias=config.bias)
        self.attn = MultiheadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_e, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None, is_causal=False):
        if self.config.pre_layernorm:
            x = x + self.attn(self.ln_1(x), attn_mask=attn_mask, is_causal=is_causal)
            x = x + self.mlp(self.ln_2(x))
        else:
            x = self.ln_1(x + self.attn(x, attn_mask=attn_mask, is_causal=is_causal))
            x = self.ln_2(x + self.mlp(x))
        return x


class LamoEncoder(nn.Module):

    def __init__(self, config: LamoConfig):

        super().__init__()
        self.config = config

        wte = nn.Embedding(config.n_vocab, config.d_e, padding_idx=config.padding_idx)

        if config.pos_type in ("none", "alibi"):
            self.encoder = nn.ModuleDict(
                {
                    "wte": wte,
                    "dropout": nn.Dropout(config.dropout),
                    "blocks": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                    "ln_f": nn.LayerNorm(config.d_e, bias=config.bias),
                }
            )
        elif config.pos_type == "learned":
            wpe = nn.Embedding(config.l_max, config.d_e)
            self.encoder = nn.ModuleDict(
                {
                    "wte": wte,
                    "wpe": wpe,
                    "dropout": nn.Dropout(config.dropout),
                    "blocks": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                    "ln_f": nn.LayerNorm(config.d_e, bias=config.bias),
                }
            )

        self.lm_head = nn.Linear(config.d_e, config.n_vocab, bias=False)
        if config.tie_weights:
            self.encoder.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("w_o.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))


    def forward(self, input_ids, attention_mask):

        bs, l_x = input_ids.shape
        assert attention_mask.shape == (bs, l_x)
        assert l_x <= self.config.l_max

        tok_emb = self.encoder.wte(input_ids)
        if self.config.pos_type in ("none", "alibi"):
            emb = tok_emb
        elif self.config.pos_type == "learned":
            pos = torch.arange(0, l_x, dtype=torch.long, device=input_ids.device)
            pos_emb = self.encoder.wpe(pos)
            emb = tok_emb + pos_emb

        x = self.encoder.dropout(emb)
        for block in self.encoder.blocks:
            x = block(x, attention_mask)
        x = self.encoder.ln_f(x)
        logits = self.lm_head(x)

        return {
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

    def get_should_decay_by_name(self):

        def should_decay(name):
            return not (
                name.endswith("bias") or
                (".ln" in name) or
                ("wte" in name) or
                ("wpe" in name)
            )
        return [(n, should_decay(n)) for n, p in self.named_parameters()]

    def get_optimizer_param_groups(self, weight_decay: float):

        name_decay_bools = self.get_should_decay_by_name()
        decay_names = set([n for n,b in name_decay_bools if b])
        no_decay_names = set([n for n,b in name_decay_bools if not b])
        all_names = set([n for n,p in self.named_parameters()])

        # check we have every parameter in some group
        chk1 = all_names - set.union(decay_names, no_decay_names)
        assert len(chk1) == 0

        # check no parameter is in both groups
        chk2 = set.intersection(decay_names, no_decay_names)
        assert len(chk2) == 0

        no_decay_params = [
            p for n, p in self.named_parameters()
            if n in no_decay_names
        ]
        decay_params = [
            p for n, p in self.named_parameters()
            if n in decay_names
        ]
        return [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]



if __name__ == "__main__":

    n_head = 12
    d_e = n_head * 64
    fc_mult = 4
    n_vocab = 2**15
    pre_layernorm = True

    config = LamoConfig(
        n_layer=12,
        n_vocab=n_vocab,
        n_head=n_head,
        l_max=512,
        d_e=d_e,
        fc_mult=fc_mult,
        pre_layernorm=pre_layernorm,
        padding_idx=0,
        dropout=0.0,
        bias=False,
        tie_weights=True,
    )
    lamo = LamoEncoder(config)

