from accelerate import Accelerator
import einops
import torch
from torch.nn import functional as F

import config_mod
import data_mod
from layers import LamoEncoder


accelerator = Accelerator(log_with="wandb")
device = accelerator.device


# RTX 3090 bf16
config = config_mod.distilbert_base_config
batch_size = 32

# RTX 3090 fp32
#config = distilbert_base_config
#batch_size = 32

# RTX 3090 bf16
#config = bert_large_config
#batch_size = 16

learning_rate = 5e-5
model = LamoEncoder(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_dataloader, val_dataloader = data_mod.get_dataloaders(batch_size, batch_size)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, scheduler
)


accelerator.init_trackers(
    project_name="lamo",
    config=config.model_dump(),
)
model.train()
for batch in train_dataloader:
    optimizer.zero_grad()
    outputs = model(batch["input_ids"], batch["attention_mask"])
    logits = einops.rearrange(outputs["logits"], "bs lx nv -> (bs lx) nv")
    labels = einops.rearrange(batch["labels"], "bs lx -> (bs lx)")
    loss = F.cross_entropy(logits, labels)
    accelerator.log({"train-loss":loss})
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
accelerator.end_training()
