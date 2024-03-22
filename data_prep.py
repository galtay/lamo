from transformers import AutoTokenizer
from datasets import load_dataset


def get_tokens(samples):
    return tokenizer(
        samples["text"],
        padding=False,
        truncation=False,
        add_special_tokens=False,
        return_length=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )


def chunk_samples(samples):
    chunks = []
    attn = []
    for ids in samples["input_ids"]:
        chunks1 = [
            [tokenizer.cls_token_id]
            + ids[i : i + block_size - 2]
            + [tokenizer.sep_token_id]
            for i in range(0, len(ids), block_size - 2)
        ]
        chunks1 = [
            seq + [tokenizer.pad_token_id] * (block_size - len(seq)) for seq in chunks1
        ]
        chunks += chunks1
        attn1 = [
            [1 if el != tokenizer.pad_token_id else 0 for el in seq] for seq in chunks1
        ]
        attn += attn1
    return {"input_ids": chunks, "attention_mask": attn}


if __name__ == "__main__":

    block_size = 2048
    tokenizer_name = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = block_size
    dsd = load_dataset("hyperdemocracy/usc-llm-text")

    # tokenizer everything with no special tokens
    dsd_tokenized = dsd.map(get_tokens, batched=True, num_proc=16)

    # break into chunks of block_size and add padding
    dsd_chunked = dsd_tokenized.map(
        chunk_samples,
        batched=True,
        num_proc=16,
        remove_columns=dsd_tokenized["train"].column_names,
    )

    repo_id = (
        "hyperdemocracy/usc-llm-tokens-"
        + tokenizer_name.split("/")[-1]
        + "-"
        + str(block_size)
    )
    dsd_chunked.push_to_hub(repo_id=repo_id)
