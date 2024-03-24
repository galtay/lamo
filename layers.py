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
    pos_type: PosType = Field(default="learned", description="type of position embeddings")
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

    def get_weight_decay_names(self):
        return [
            n for n, p in self.named_parameters()
            if not (n.endswith("bias") or ".ln" in n)
        ]

    def get_no_weight_decay_names(self):
        return [
            n for n, p in self.named_parameters()
            if (n.endswith("bias") or ".ln" in n)
        ]

    def get_optimizer_param_groups(self, weight_decay: float):
        no_decay_params = [
            p for n, p in self.named_parameters()
            if n in self.get_no_weight_decay_names()
        ]
        decay_params = [
            p for n, p in self.named_parameters()
            if n in self.get_weight_decay_names()
        ]
        return [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]



if __name__ == "__main__":

    d_e = 768
    n_head = 12
    d_mid = d_e // n_head
    d_attn = d_e // n_head
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

    config_formal = FormalLamoConfig(
        n_layer=12,
        n_vocab=n_vocab,
        n_head=n_head,
        l_max=512,
        d_x=d_e,
        d_z=d_e,
        d_attn=n_head * 4,
        d_mid=n_head * 5,
        d_out=n_head * 6,
        fc_mult=fc_mult,
        pre_layernorm=pre_layernorm,
        padding_idx=0,
        dropout=0.0,
        bias=False,
        tie_weights=True,
    )


    from data_mod import get_dataloaders
    from flash_attn.bert_padding import unpad_input

    dataloaders = get_dataloaders(32, 32)
    train_dl = dataloaders['train_dl']
    val_dl = dataloaders['val_dl']
    batch = next(iter(train_dl))
    lamo = LamoEncoder(config)

    hidden_states = lamo.encoder.wte(batch['input_ids'])
    attn_mask = batch['attention_mask']

    u_hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(hidden_states, attn_mask)

    out = lamo(batch['input_ids'], batch['attention_mask'])

    sys.exit(0)



    mha = MultiheadAttention(config)
    fmha = FormalMultiheadAttention(config_formal)

    b_s = 3
    l_x = config.l_max
    l_z = config.l_max
    x = torch.rand(b_s, l_x, config_formal.d_x)
    z = torch.rand(b_s, l_z, config_formal.d_z)
    mha_out = mha(x)
    fmha_out = fmha(x, z)

    input_ids = torch.randint(0, n_vocab, size=(b_s, l_x))
    attention_mask = torch.ones_like(input_ids)
    lamo = LamoEncoder(config)
    rich.print("number of parameters: %.4fM" % (lamo.get_num_params() / 1e6,))
    lamo_out = lamo(input_ids, attention_mask)
