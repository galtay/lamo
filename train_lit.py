import einops
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import rich
import torch
from torch.nn import functional as F

import config_mod
import data_mod
from layers import LamoEncoder


seed = 8272
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

batch_size = 16
learning_rate = 5e-5
weight_decay = 0.01
l_max = 2048


# RTX 3090 bf16
config = config_mod.distilbert_base_config
#config.attn_impl = "torch_spda"
config.attn_impl = "flash_varlen_qkvpacked"
config.pos_type = "alibi"
#config.pos_type = "learned"
#config.pos_type = "none"
config.l_max = l_max
rich.print(config)


torch.set_float32_matmul_precision("high")
wandb_logger = WandbLogger(project="lamo")

class LitLamo(L.LightningModule):

    def __init__(self, config, learning_rate, weight_decay):
        super().__init__()
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = LamoEncoder(config)
        if config.attn_impl != "flash_varlen_qkvpacked":
            rich.print("compiling model")
            self.model = torch.compile(self.model)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch["input_ids"], batch["attention_mask"])
        logits = einops.rearrange(outputs["logits"], "bs lx nv -> (bs lx) nv")
        labels = einops.rearrange(batch["labels"], "bs lx -> (bs lx)")
        loss = F.cross_entropy(logits, labels)
        self.log("train-loss", loss)
        return loss

    def configure_optimizers(self):
        optim_groups = self.model.get_optimizer_param_groups(self.weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, fused=True)
        return optimizer


lit_lamo = LitLamo(config, learning_rate, weight_decay)
dataloaders = data_mod.get_dataloaders(batch_size, batch_size, l_max, num_workers=2)
trainer = L.Trainer(
    limit_train_batches=512,
    max_epochs=1,
    precision="bf16-mixed",
    devices=2,
    logger=wandb_logger,
)
trainer.fit(model=lit_lamo, train_dataloaders=dataloaders['train_dl'])
