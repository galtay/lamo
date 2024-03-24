"""
accelerate env

accelerate launch train_acc.py

"""

from accelerate import Accelerator
import einops
import torch
from torch.nn import functional as F
from tqdm import tqdm

import config_mod
import data_mod
from layers import LamoEncoder


if __name__ == "__main__":

    accelerator = Accelerator(log_with="wandb")
    device = accelerator.device


    # RTX 3090 bf16
    config = config_mod.distilbert_base_config
    config.attn_impl = "torch_spda"
    config.pos_type = "learned"
    batch_size = 32

    # RTX 3090 fp32
    #config = distilbert_base_config
    #batch_size = 32

    # RTX 3090 bf16
    #config = bert_large_config
    #batch_size = 16

    weight_decay = 0.01
    learning_rate = 5e-5
    model = LamoEncoder(config)
    optimizer = torch.optim.AdamW(model.get_optimizer_param_groups(weight_decay), lr=learning_rate)
    dataloaders = data_mod.get_dataloaders(batch_size, batch_size, config.l_max)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloaders["train_dl"], dataloaders["val_dl"], scheduler
    )

    accelerator.init_trackers(
        project_name="lamo",
        config=config.model_dump(),
    )
    model.train()
    epoch = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"train: {epoch=}")):
        outputs = model(batch["input_ids"], batch["attention_mask"])
        logits = einops.rearrange(outputs["logits"], "bs lx nv -> (bs lx) nv")
        labels = einops.rearrange(batch["labels"], "bs lx -> (bs lx)")
        loss = F.cross_entropy(logits, labels)
        accelerator.log({"train-loss":loss})
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    accelerator.end_training()
