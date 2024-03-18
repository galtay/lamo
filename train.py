from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import wandb

from layers import Malamo, MalamoConfig

n_layer = 12 // 2
batch_size = 32
num_tokens = 512
d_embed = 512 // 2
num_heads = 8
bias = False
dropout = 0.0
learning_rate = 5e-5
weight_decay = 0.0
device = torch.device("cuda")

tokenizer_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

config = MalamoConfig(
    n_layer=n_layer,
    vocab_size=tokenizer.vocab_size,
    d_embed=d_embed,
    num_heads=num_heads,
    padding_idx=tokenizer.pad_token_id,
    dropout=dropout,
    bias=bias,
    max_seq_len=num_tokens,
)


dsd = load_dataset("hyperdemocracy/usc-llm-tokens-bert-base-uncased-512")
dsd.set_format("torch")

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

train_dataloader = DataLoader(
    dsd["train"], shuffle=True, collate_fn=collator, batch_size=batch_size
)
val_dataloader = DataLoader(
    dsd["validation"], shuffle=False, collate_fn=collator, batch_size=batch_size
)

model = Malamo(config).to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

wandb.init(
    project="malamo",
    config=config.model_dump(),
)

epochs = 4
log_every = 20
valid_every = 500
global_step = 0
loss_buffer = []


for epoch in range(epochs):

    for ii_batch_step, train_batch in enumerate(train_dataloader):

        train_batch = {k:v.to(device) for k,v in train_batch.items()}
        optimizer.zero_grad()
        outputs = model(**train_batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        loss_buffer.append(loss.item())

        if global_step % log_every == 0 and global_step != 0:
            avg_loss = sum(loss_buffer) / len(loss_buffer)
            wandb.log(data={"train_loss": avg_loss}, step=global_step, commit=False)
            print("global_step: {}, loss: {}".format(global_step, avg_loss))
            loss_buffer = []

        if global_step % valid_every == 0 and global_step != 0:
            running_loss = 0
            model.eval()
            with torch.no_grad():
                for ii_val_step, valid_batch in enumerate(val_dataloader):
                    valid_batch = {k:v.to(device) for k,v in valid_batch.items()}
                    outputs = model(**valid_batch)
                    loss = outputs["loss"]
                    running_loss += loss.item()
                avg_vloss = running_loss / (ii_val_step+1)
            wandb.log(data={"val_loss": avg_vloss}, step=global_step, commit=False)
            model.train()

        wandb.log(data={}, commit=True)
        global_step += 1

wandb.finish()








