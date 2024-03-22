from transformers import AutoTokenizer
from layers import LamoConfig


tokenizer_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

d_embed = 768
n_head = 12
distilbert_base_config = LamoConfig(
    n_layer=6,
    n_vocab=tokenizer.vocab_size,
    n_head=n_head,
    l_max=512,
    d_x=d_embed,
    d_z=d_embed,
    d_attn=d_embed//n_head,
    d_mid=d_embed//n_head,
    d_out=d_embed,
    padding_idx=0,
    dropout=0.1,
    bias=False,
)

d_embed = 768
n_head = 12
distilbert_base_long_config = LamoConfig(
    n_layer=6,
    n_vocab=tokenizer.vocab_size,
    n_head=n_head,
    l_max=2048,
    d_x=d_embed,
    d_z=d_embed,
    d_attn=d_embed//n_head,
    d_mid=d_embed//n_head,
    d_out=d_embed,
    padding_idx=0,
    dropout=0.1,
    bias=False,
)

d_embed = 768
n_head = 12
bert_base_config = LamoConfig(
    n_layer=12,
    n_vocab=tokenizer.vocab_size,
    n_head=n_head,
    l_max=512,
    d_x=d_embed,
    d_z=d_embed,
    d_attn=d_embed//n_head,
    d_mid=d_embed//n_head,
    d_out=d_embed,
    padding_idx=0,
    dropout=0.1,
    bias=False,
)

d_embed = 1024
n_head = 16
bert_large_config = LamoConfig(
    n_layer=24,
    n_vocab=tokenizer.vocab_size,
    n_head=n_head,
    l_max=512,
    d_x=d_embed,
    d_z=d_embed,
    d_attn=d_embed//n_head,
    d_mid=d_embed//n_head,
    d_out=d_embed,
    padding_idx=0,
    dropout=0.1,
    bias=False,
)
