import einops
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import rich
import torch
from torch.nn import functional as F

import config_mod
import data_mod
from layers import LamoEncoder


# RTX 3090 bf16
#config = config_mod.distilbert_base_config
#batch_size = 64

# RTX 3090 bf16
config = config_mod.distilbert_base_long_config
config.pos_emb_type = "none"
batch_size = 32


# RTX 3090 fp32
#config = distilbert_base_config
#batch_size = 32

# RTX 3090 bf16
#config = bert_large_config
#batch_size = 16

learning_rate = 5e-5
weight_decay = 0.01
torch.set_float32_matmul_precision("high")
wandb_logger = WandbLogger(project="lamo")

class LitLamo(L.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.model = LamoEncoder(config)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch["input_ids"], batch["attention_mask"])
        logits = einops.rearrange(outputs["logits"], "bs lx nv -> (bs lx) nv")
        labels = einops.rearrange(batch["labels"], "bs lx -> (bs lx)")
        loss = F.cross_entropy(logits, labels)
        self.log("train-loss", loss)
        return loss

    def configure_optimizers(self):
        nodecay_params = [
            p for n, p in lit_lamo.named_parameters()
            if (n.endswith("bias") or ".ln" in n)
        ]
        decay_params = [
            p for n, p in lit_lamo.named_parameters()
            if not (n.endswith("bias") or ".ln" in n)
        ]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=True)
        return optimizer

rich.print(config)
lit_lamo = LitLamo(config)
train_dataloader, val_dataloader = data_mod.get_dataloaders(batch_size, batch_size, num_workers=2)
trainer = L.Trainer(
#    limit_train_batches=100,
    max_epochs=1,
    precision="bf16-mixed",
    devices=2,
    logger=wandb_logger,
)
trainer.fit(model=lit_lamo, train_dataloaders=train_dataloader)
