from typing import Any, List
from pytorch_lightning.core import optimizer
import torch

import pytorch_lightning as pl
from torch import nn
from torchmetrics.aggregation import MinMetric
from torchmetrics.regression.mean_squared_error import MeanSquaredError

from src.models.autoencoder import Autoencoder


class AutoencoderRegressor(pl.LightningModule):
    def __init__(
        self,
        #feature_extractor: Autoencoder,
        ckpt_path: str,  # TODO: think of another name and perhaps way of achieving the loading of the autoencoder from a checkpoint 
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.feature_extractor = Autoencoder.load_from_checkpoint(ckpt_path)
        self.feature_extractor.freeze()
        
        # TODO: think of something better to get latent_size
        latent_size = self.feature_extractor.encoder.latent_size
        
        # TODO: make the regressor bigger, ideally create a distinct MLP module which is reusable
        self.regressor = nn.Linear(latent_size, 1)

        self.criterion = torch.nn.MSELoss()

        self.criterion = torch.nn.MSELoss()

        self.train_mse = MeanSquaredError() 
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.val_mse_best = MinMetric()

    def forward(self, x: torch.Tensor):
        _, embedding = self.feature_extractor(x)
        y_hat = self.regressor(embedding)

        return y_hat

    def _get_loss(self, batch, stage):
        if stage == "fit":
            x, y = batch["labeled"]
        
        if stage in ("val", "test"):
            x, y = batch
        
        y = y.view(-1, 1)
        y_hat = self.forward(x)
        loss = self.criterion(y, y_hat)

        return loss, y, y_hat

    def training_step(self, batch: Any, batch_idx: int):
        
        loss, y, y_hat = self._get_loss(batch, stage="fit")

        mse = self.train_mse(y, y_hat)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/mse", mse, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, y, y_hat = self._get_loss(batch, stage="val")

        mse = self.val_mse(y, y_hat)
        self.log("val/loss", loss)
        self.log("val/mse", mse)

    def validation_epoch_end(self, outputs: List[Any]):
        mse = self.val_mse.compute()
        self.val_mse_best.update(mse)
        self.log("val/mse_best", self.val_mse_best.compute(),on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        loss, y, y_hat = self._get_loss(batch, stage="test")

        mse = self.test_mse(y, y_hat)
        self.log("test/loss", loss)
        self.log("test/mse", mse)
    
    def on_epoch_end(self) -> None:
        self.train_mse.reset()
        self.val_mse.reset()
        self.test_mse.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr = self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer