from omegaconf import OmegaConf

from chiclab.roll.trainer import ChicTrainer
from chiclab.roll.optim import build_optimizer, build_scheduler

from src.model import TwitterModel
from src.data import TwitterDataModule

class TwitterTrainer(ChicTrainer):
    def configure_optimizers(self, model):
        params = [p for p in model.parameters() if p.requires_grad] + \
            [p for p in self.nest.parameters() if p.requires_grad]
        optimizer = build_optimizer(
            "adamw", params, **self.optimizer
        )

        scheduler = build_scheduler(
            "sequential", optimizer, self.scheduler
        )

        return optimizer, scheduler

    def train_step(self, batch):
        loss, loss_part = self.nest.compute_loss(**batch)
        return loss, loss_part
    
    def valid_step(self, batch):
        loss, loss_part = self.nest.compute_loss(**batch)
        return loss, loss_part

    def log_step(self):
        self.writer.add_scalar("train/loss", self.log_values["train/loss"], self.log_values["global_step"])
        self.writer.add_scalar("train/loss_ams", self.log_values["train/metrics"]["ams"], self.log_values["global_step"])
        self.writer.add_scalar("train/loss_supcon", self.log_values["train/metrics"]["supcon"], self.log_values["global_step"])
        if self.log_values["global_step"] % self.valid_per_steps == 0:
            self.writer.add_scalar("valid/loss", self.log_values["valid/loss"], self.log_values["global_step"])
            self.writer.add_scalar("valid/loss_ams", self.log_values["valid/metrics"]["ams"], self.log_values["global_step"])
            self.writer.add_scalar("valid/loss_supcon", self.log_values["valid/metrics"]["supcon"], self.log_values["global_step"])

def main(config):
    model = TwitterModel(**config.model)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)

    datamodule = TwitterDataModule(config.data.dataset, config.data.tokenizer, config.data.tab)
    datamodule.prepare()

    trainer = TwitterTrainer(**config.trainer)
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    import sys
    config = OmegaConf.load(sys.argv[1])
    main(config)