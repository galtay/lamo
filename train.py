"""
https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
https://pytorch.org/docs/master/generated/torch.set_float32_matmul_precision.html?highlight=precision#torch.set_float32_matmul_precision
https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html


TODO: add weight_decay
TODO: checkpoints

"""

from datasets import load_dataset
import rich
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import wandb

from layers import LamoEncoder, LamoConfig


TORCH_FLOAT32_MATMUL_PRECISIONS = ["highest", "high", "medium"]


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
    dropout=0.0,
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
    dropout=0.0,
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
    dropout=0.0,
    bias=False,
)


def get_dataloaders(train_batch_size, val_batch_size, mlm_probability=0.15):

    dsd = load_dataset("hyperdemocracy/usc-llm-tokens-bert-base-uncased-512")
    dsd.set_format("torch")
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )
    train_dataloader = DataLoader(
        dsd["train"], shuffle=True, collate_fn=collator, batch_size=train_batch_size
    )
    val_dataloader = DataLoader(
        dsd["validation"], shuffle=False, collate_fn=collator, batch_size=val_batch_size
    )
    return train_dataloader, val_dataloader


def run_val_loop(model, dataloader, device, use_amp, amp_dtype):
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="eval")):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(**batch)
            loss = outputs["loss"]
            running_loss += loss.item()
    avg_loss = running_loss / (step + 1)
    return avg_loss


def run_train_epoch(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    global_step,
    log_every,
    val_every,
    device,
    use_amp,
    amp_dtype,
):

    loss_buffer = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for step, batch in enumerate(tqdm(train_dataloader, desc=f"train: {epoch=}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(**batch)
            loss = outputs["loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        loss_buffer.append(loss.item())

        if global_step % log_every == 0 and global_step != 0:
            avg_loss = sum(loss_buffer) / len(loss_buffer)
            wandb.log(data={"train_loss": avg_loss}, step=global_step, commit=False)
            #print("global_step: {}, loss: {}".format(global_step, avg_loss))
            loss_buffer = []

        if global_step % val_every == 0 and global_step != 0:
            avg_val_loss = run_val_loop(model, val_dataloader, device, use_amp, amp_dtype)
            wandb.log(data={"val_loss": avg_val_loss}, step=global_step, commit=False)
            model.train()

        wandb.log(data={}, commit=True)
        global_step += 1

    return global_step


# RTX 3090 bf16
config = distilbert_base_config
batch_size = 64

# RTX 3090 fp32
#config = distilbert_base_config
#batch_size = 32

# RTX 3090 bf16
#config = bert_large_config
#batch_size = 16

learning_rate = 5e-5
weight_decay = 0.0
epochs = 1
log_every = 20
val_every = 500
torch_float32_matmul_precision = (
    "high"  # set to "high" or "medium" to enable TensorFloat32 (TF32) mode
)
use_amp = True
amp_dtype = torch.bfloat16

torch.set_float32_matmul_precision(torch_float32_matmul_precision)
device = torch.device("cuda:1")

train_dataloader, val_dataloader = get_dataloaders(batch_size, batch_size)
tokens_per_step = batch_size * config.l_max
rich.print(f"{tokens_per_step=}")

model = LamoEncoder(config).to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
wandb.init(
    project="malamo",
    config=config.model_dump(),
)

model.train()
global_step = 0
for epoch in range(epochs):
    global_step = run_train_epoch(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        global_step,
        log_every,
        val_every,
        device,
        use_amp,
        amp_dtype,
    )
wandb.finish()
