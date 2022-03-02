from typing import Any, List
import pytorch_lightning as pl

from torchmetrics import MetricCollection, MetricTracker, MeanSquaredError

import torch


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        lr: float = 0.001,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.encoder = encoder
        self.decoder = decoder

        self.criterion = torch.nn.MSELoss()

        metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(),
                "RMSE": MeanSquaredError(squared=False),
            }
        )

        self.train_metrics = MetricTracker(metrics.clone(prefix='train/'), maximize=[False, False])
        self.valid_metrics = MetricTracker(metrics.clone(prefix='val/'), maximize=[False, False])
        self.test_metrics = MetricTracker(metrics.clone(prefix='test/'), maximize=[False, False])
        #self.train_mse = MeanSquaredError() 
        #self.val_mse = MeanSquaredError()
        #self.test_mse = MeanSquaredError()

        #self.val_mse_best = MinMetric()

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    #def _get_reconstruction_loss(self, batch, stage):

        #if stage == "fit":
            #x_labeled, _ = batch["labeled"]
            #x_unlabeled, _ = batch["unlabeled"]

            #x = torch.cat((x_labeled, x_unlabeled), dim=0)
        
        #if stage in ("val", "test"):
            #x, _ = batch

        #x_hat, _ = self.forward(x)
        #loss = self.criterion(x, x_hat)

        #return loss, x, x_hat


    def on_train_epoch_start(self) -> None:
        self.train_metrics.increment()

    def training_step(self, batch: Any, batch_idx: int):

        x_labeled, _ = batch["labeled"]
        x_unlabeled, _ = batch["unlabeled"]

        x = torch.cat((x_labeled, x_unlabeled), dim=0)
        
        x_hat, _ = self.forward(x)
        loss = self.criterion(x, x_hat)

        # Log the loss
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        self.train_metrics(x_hat, x)

        return loss

    def training_epoch_end(self, outputs):
        # Log the metrics at the end of each training epoch
        self.log_dict(self.train_metrics.compute())


    def on_validation_epoch_start(self):
        self.valid_metrics.increment()

    def validation_step(self, batch: Any, batch_idx: int):
        x, _ = batch
        x_hat, _ = self.forward(x)

        self.valid_metrics(x_hat, x)

    def validation_epoch_end(self, outputs: List[Any]):
        self.log_dict(self.valid_metrics.compute())
        best_metrics, _ = self.valid_metrics.best_metric(return_step=True)
        best_metrics = {f"{key}_best": val for key, val in best_metrics.items()}
        self.log_dict(best_metrics)


    def on_test_epoch_start(self) -> None:
        self.test_metrics.increment()

    def test_step(self, batch: Any, batch_idx: int):
        x, _ = batch
        x_hat, _ = self.forward(x)

        self.test_metrics(x_hat, x)

    def test_epoch_end(self, outputs: List[Any]):
        self.log_dict(self.test_metrics.compute())


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
        )
        return optimizer