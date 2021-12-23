from typing import Any, List
import pytorch_lightning as pl
from torchmetrics.aggregation import MinMetric

from torchmetrics.regression.mean_squared_error import MeanSquaredError

import torch

from src.models.modules.encoder import Encoder
from src.models.modules.decoder import Decoder

class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        #input_size: int = 18,
        #lin1_size: int = 256,
        #lin2_size: int = 128,
        #lin3_size: int = 64,
        #latent_size: int = 32,
        #lin4_size: int = 64,
        #lin5_size: int = 128,
        #lin6_size: int = 256,
        #output_size: int = 18,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.encoder = encoder
        self.decoder = decoder
        #self.encoder = Encoder(hparams=self.hparams)
        #self.decoder = Decoder(hparams=self.hparams)

        self.criterion = torch.nn.MSELoss()

        self.train_mse = MeanSquaredError() 
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.val_mse_best = MinMetric()

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def _get_reconstruction_loss(self, batch, stage):

        if stage == "fit":
            x_labeled, _ = batch["labeled"]
            x_unlabeled, _ = batch["unlabeled"]

            x = torch.cat((x_labeled, x_unlabeled), dim=0)
        
        if stage in ("val", "test"):
            x, _ = batch

        x_hat, _ = self.forward(x)
        loss = self.criterion(x, x_hat)

        return loss, x, x_hat

    def training_step(self, batch: Any, batch_idx: int):
        loss, x, x_hat = self._get_reconstruction_loss(batch, stage="fit")

        mse = self.train_mse(x, x_hat)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/mse", mse, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, x, x_hat = self._get_reconstruction_loss(batch, stage="val")

        mse = self.val_mse(x, x_hat)
        self.log("val/loss", loss)
        self.log("val/mse", mse)

    def validation_epoch_end(self, outputs: List[Any]):
        mse = self.val_mse.compute()
        self.val_mse_best.update(mse)
        self.log("val/mse_best", self.val_mse_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, x, x_hat = self._get_reconstruction_loss(batch, stage="test")

        mse = self.test_mse(x, x_hat)
        self.log("test/loss", loss)
        self.log("test/mse", mse)

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self) -> None:
        self.train_mse.reset()
        self.val_mse.reset()
        self.test_mse.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer