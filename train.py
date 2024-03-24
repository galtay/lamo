"""
https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
https://pytorch.org/docs/master/generated/torch.set_float32_matmul_precision.html?highlight=precision#torch.set_float32_matmul_precision
https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html



TODO: checkpoints

"""

import rich
import einops
import torch
from torch.nn import functional as F
from tqdm import tqdm
import wandb

import config_mod
import data_mod
from layers import LamoEncoder


TORCH_FLOAT32_MATMUL_PRECISIONS = ["highest", "high", "medium"]


def run_val_loop(model, dataloader, device, use_amp, amp_dtype):
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="eval")):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(batch["input_ids"], batch["attention_mask"])
                logits = einops.rearrange(outputs["logits"], "bs lx nv -> (bs lx) nv")
                labels = einops.rearrange(batch["labels"], "bs lx -> (bs lx)")
                loss = F.cross_entropy(logits, labels)
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
    scaler,
    max_steps=None,
):

    loss_buffer = []

    for step, batch in enumerate(tqdm(train_dataloader, desc=f"train: {epoch=}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(batch["input_ids"], batch["attention_mask"])
            bs, lx, n_vocab = outputs["logits"].shape
            logits = einops.rearrange(outputs["logits"], "bs lx nv -> (bs lx) nv")
            labels = einops.rearrange(batch["labels"], "bs lx -> (bs lx)")
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        loss_buffer.append(loss.item())

        if global_step % log_every == 0 and global_step != 0:
            avg_loss = sum(loss_buffer) / len(loss_buffer)
            wandb.log(data={"train-loss": avg_loss}, step=global_step, commit=False)
            loss_buffer = []

        if global_step % val_every == 0 and global_step != 0:
            avg_val_loss = run_val_loop(model, val_dataloader, device, use_amp, amp_dtype)
            wandb.log(data={"val-loss": avg_val_loss}, step=global_step, commit=False)
            model.train()

        wandb.log(data={}, commit=True)
        global_step += 1

        if max_steps is not None and global_step >= max_steps:
            break

    return global_step


def get_optimizer(model, learning_rate, weight_decay=0.0):
    no_decay_params = [
        p for n, p in model.named_parameters()
        if n in model.get_no_weight_decay_names()
    ]
    decay_params = [
        p for n, p in model.named_parameters()
        if n in model.get_weight_decay_names()
    ]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=True)
    return optimizer


# RTX 3090 bf16
#config = config_mod.distilbert_base_config
#batch_size = 64

# RTX 3090 bf16
config = config_mod.distilbert_base_long_config
batch_size = 32

# RTX 3090 fp32
#config = distilbert_base_config
#batch_size = 32

# RTX 3090 bf16
#config = bert_large_config
#batch_size = 16

rich.print(config)
learning_rate = 5e-5
weight_decay = 0.01
epochs = 1
log_every = 20
val_every = 5000000
max_steps = 5000
#max_steps = None
torch_float32_matmul_precision = (
    "high"  # set to "high" or "medium" to enable TensorFloat32 (TF32) mode
)
use_amp = True
amp_dtype = torch.bfloat16

torch.set_float32_matmul_precision(torch_float32_matmul_precision)
device = torch.device("cuda:0")

dataloaders = data_mod.get_dataloaders(batch_size, batch_size)
tokens_per_step = batch_size * config.l_max
rich.print(f"{tokens_per_step=}")

model = LamoEncoder(config).to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.get_optimizer_param_groups(weight_decay), lr=learning_rate, fused=True)

wandb.init(
    project="lamo",
    config=config.model_dump(),
)

model.train()
global_step = 0
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
for epoch in range(epochs):
    global_step = run_train_epoch(
        model,
        optimizer,
        dataloaders["train_dl"],
        dataloaders["val_dl"],
        global_step,
        log_every,
        val_every,
        device,
        use_amp,
        amp_dtype,
        scaler,
        max_steps=max_steps,
    )
    torch.save({
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }, "output.pt")

wandb.finish()
