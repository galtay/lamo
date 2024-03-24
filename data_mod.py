from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling


def get_dataloaders(
    train_batch_size, val_batch_size, max_seq_len, mlm_probability=0.15, num_workers=1
):

    assert max_seq_len in (512, 1024, 2048)
    tokenizer_name = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dsd = load_dataset(f"hyperdemocracy/usc-llm-tokens-bert-base-uncased-{max_seq_len}")
    dsd.set_format("torch")
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )
    train_dataloader = DataLoader(
        dsd["train"],
        shuffle=True,
        collate_fn=collator,
        batch_size=train_batch_size,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        dsd["validation"],
        shuffle=False,
        collate_fn=collator,
        batch_size=val_batch_size,
        num_workers=num_workers,
    )
    return {
        "train_dl": train_dataloader,
        "val_dl": val_dataloader,
        "tokenizer": tokenizer,
    }
