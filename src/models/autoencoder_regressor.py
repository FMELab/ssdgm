from typing import Any, List
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from torch import nn

from torchmetrics import MetricCollection, MetricTracker, MeanSquaredError

from src.models.autoencoder import Autoencoder
from src.models.modules.dense import Fcn


class AutoencoderRegressor(pl.LightningModule):
    def __init__(
        self,
        #feature_extractor: Autoencoder,
        checkpoint_path: str,  
        regressor: Fcn,
        lr: float = 0.001,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.feature_extractor = Autoencoder.load_from_checkpoint(checkpoint_path)
        self.feature_extractor.freeze()
        
        self.regressor = regressor


        metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(),
                "RMSE": MeanSquaredError(squared=False),
            }
        )

        self.train_metrics = MetricTracker(metrics.clone(prefix='train/'), maximize=[False, False])
        self.valid_metrics = MetricTracker(metrics.clone(prefix='val/'), maximize=[False, False])
        self.test_metrics = MetricTracker(metrics.clone(prefix='test/'), maximize=[False, False])

    def forward(self, x: torch.Tensor):
        _, embedding = self.feature_extractor(x)
        y_hat = self.regressor(embedding)

        return y_hat

    #def _get_loss(self, batch, stage):
        #if stage == "fit":
            #x, y = batch["labeled"]
        
        #if stage in ("val", "test"):
            #x, y = batch
        
        #y = y.view(-1, 1)
        #y_hat = self.forward(x)
        #loss = self.criterion(y, y_hat)

        #return loss, y, y_hat

    def on_train_epoch_start(self) -> None:
        self.train_metrics.increment()

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch["labeled"]

        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        self.train_metrics(y_hat, y)

        return loss

    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics.compute())


    def on_validation_epoch_start(self):
        self.valid_metrics.increment()

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch

        y_hat = self.forward(x)

        self.valid_metrics(y_hat, y)

    def validation_epoch_end(self, outputs: List[Any]):
        self.log_dict(self.valid_metrics.compute())
        best_metrics, _ = self.valid_metrics.best_metric(return_step=True)
        best_metrics = {f"{key}_best": val for key, val in best_metrics.items()}
        self.log_dict(best_metrics)


    def on_test_epoch_start(self) -> None:
        self.test_metrics.increment()        

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch

        y_hat = self.forward(x)

        self.test_metrics(y_hat, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr = self.hparams.lr,
        )
        return optimizer