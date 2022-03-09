import pytorch_lightning as pl
import torch as T
import torch.nn.functional as F

from src.models.modules.dense import Fcn
from torchmetrics import MetricCollection, MetricTracker, MeanSquaredError

class MultiLayerPerceptron(pl.LightningModule):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        leaky_relu_slope,
        dropout_proba,
        lr,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.net = Fcn(
            in_features,
            hidden_features,
            out_features,
            leaky_relu_slope,
            dropout_proba,
        )

        metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(),
                "RMSE": MeanSquaredError(squared=False),
            }
        )

        self.train_metrics = MetricTracker(metrics.clone(prefix="train/"), maximize=[False, False])
        self.valid_metrics = MetricTracker(metrics.clone(prefix="val/"), maximize=[False, False])
        self.test_metrics = MetricTracker(metrics.clone(prefix="test/"), maximize=[False, False])

    def forward(self, x):
        y_hat = self.net(x)
        return y_hat
    
    def on_train_epoch_start(self):
        self.train_metrics.increment()

    def training_step(self, batch, batch_idx):
        x_labeled, y = batch["labeled"]

        y_hat = self.forward(x_labeled)

        loss = F.mse_loss(y_hat, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        self.train_metrics(y_hat, y)

        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())

    def on_validation_epoch_start(self):
        self.valid_metrics.increment()

    def validation_step(self, batch, batch_idx):
        x_labeled, y = batch

        y_hat = self.forward(x_labeled)        
        self.valid_metrics(y_hat, y)

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute())
        best_metrics, _ = self.valid_metrics.best_metric(return_step=True)
        best_metrics = {f"{key}_best": val for key, val in best_metrics.items()}
        self.log_dict(best_metrics)
    
    def on_test_epoch_start(self):
        self.test_metrics.increment()

    def test_step(self, batch, batch_idx):
        x_labeled, y = batch

        y_hat = self.forward(x_labeled)        
        self.test_metrics(y_hat, y)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def configure_optimizers(self):
        optimizer = T.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
        )

        return optimizer