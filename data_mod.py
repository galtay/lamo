from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling


def get_dataloaders(
    train_batch_size, val_batch_size, mlm_probability=0.15, num_workers=1
):

    tokenizer_name = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dsd = load_dataset("hyperdemocracy/usc-llm-tokens-bert-base-uncased-512")
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
    return train_dataloader, val_dataloader
