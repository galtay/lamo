"""
Inspired by,
* https://github.com/karpathy/nanoGPT/tree/master
* https://theaisummer.com/einsum-attention/

Notes on padding and masking
* https://github.com/pytorch/pytorch/issues/103749

"""

import math

import einops
from pydantic import BaseModel
import torch
import torch.nn as nn
from torch.nn import functional as F


class MalamoConfig(BaseModel):
    n_layer: int
    vocab_size: int
    d_embed: int
    num_heads: int
    max_seq_len: int
    padding_idx: int
    dropout: float
    bias: bool


class MultiheadSelfAttention(nn.Module):

    def __init__(self, config: MalamoConfig):

        super().__init__()
        assert (
            config.d_embed % config.num_heads == 0
        ), "d_embed is not divisible by num_heads"

        self.config = config
        self.d_head = config.d_embed // config.num_heads
        self.resid_dropout = nn.Dropout(config.dropout)
        self.w_qkv = nn.Linear(config.d_embed, 3 * config.d_embed, bias=config.bias)
        self.w_o = nn.Linear(config.d_embed, config.d_embed, bias=config.bias)

    def forward(self, x, attn_mask=None, is_causal=False):
        """bs: batch_size, sl: sequence length, de: embedding dimension
        nh: num heads,  dh: head dimension. de = nh * dh
        """

        bs, sl, de = x.shape
        assert de == self.config.d_embed
        qkv = self.w_qkv(x)
        assert qkv.shape == (bs, sl, 3 * de)
        qq, kk, vv = tuple(
            einops.rearrange(
                qkv, "bs sl (k nh dh) -> k bs nh sl dh", k=3, nh=self.config.num_heads
            )
        )
        dropout_p = self.config.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            qq, kk, vv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )
        assert y.shape == (bs, self.config.num_heads, sl, self.d_head)
        y = einops.rearrange(y, "bs nh sl dh -> bs sl (nh dh)")
        y = self.resid_dropout(self.w_o(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.w_fc_in = nn.Linear(config.d_embed, 4 * config.d_embed, bias=config.bias)
        self.gelu = nn.GELU()
        self.w_fc_out = nn.Linear(4 * config.d_embed, config.d_embed, bias=config.bias)
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
        self.layernorm_1 = nn.LayerNorm(config.d_embed, bias=config.bias)
        self.attn = MultiheadSelfAttention(config)
        self.layernorm_2 = nn.LayerNorm(config.d_embed, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.layernorm_1(x), attn_mask)
        x = x + self.mlp(self.layernorm_2(x))
        return x


class Malamo(nn.Module):

    def __init__(self, config: MalamoConfig):

        super().__init__()
        self.config = config

        wte = nn.Embedding(
            config.vocab_size, config.d_embed, padding_idx=config.padding_idx
        )
        wpe = nn.Embedding(config.max_seq_len, config.d_embed)

        self.encoder = nn.ModuleDict(
            {
                "wte": wte,
                "wpe": wpe,
                "dropout": nn.Dropout(config.dropout),
                "blocks": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "layernorm_f": nn.LayerNorm(config.d_embed, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("w_o.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.4fM" % (self.get_num_params() / 1e6,))

    def forward(self, input_ids, attention_mask, labels=None):

        bs, sl = input_ids.shape

        assert attention_mask.shape == (bs, sl)
        assert sl <= self.config.max_seq_len

        pos = torch.arange(0, sl, dtype=torch.long, device=input_ids.device)

        # attention mask from tokenizer is [bs, sl]
        # first convert to [bs, sl, sl] by repeating each sequence sl times
        attn_mask = attention_mask.repeat(1, 1, sl).reshape(bs, sl, sl)

        # now make boolean and expand so it is broadcastable to [bs, nh, sl, sl]
        attn_mask = attn_mask == 1
        attn_mask = attn_mask[:, None, :, :]

        tok_emb = self.encoder.wte(input_ids)
        pos_emb = self.encoder.wpe(pos)
        x = self.encoder.dropout(tok_emb + pos_emb)
        for block in self.encoder.blocks:
            x = block(x, attn_mask)
        x = self.encoder.layernorm_f(x)
        logits = self.lm_head(x)

        if labels is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "attn_mask": attn_mask,
            "tok_emb": tok_emb,
            "pos_emb": pos_emb,
            "x": x,
            "logits": logits,
            "labels": labels,
            "loss": loss,
        }

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.encoder.wte.weight.numel()
        return n_params
