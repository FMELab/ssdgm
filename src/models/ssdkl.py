from src.models.modules.dkl import DeepKernelLearning

import pytorch_lightning as pl
import torch
import gpytorch

from torchmetrics import MetricCollection, MetricTracker, MeanSquaredError
from typing import Any, List


debug_list = []

class SemiSupervisedDeepKernelLearning(pl.LightningModule):
    def __init__(
        self,
        feature_extractor,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        variance_loss_multiplier=1.0,
        lr=0.01,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.likelihood = likelihood
        self.feature_extractor = feature_extractor

        metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(),
                "RMSE": MeanSquaredError(squared=False),
            }
        )

        self.train_metrics = MetricTracker(metrics.clone(prefix='train/'), maximize=[False, False])
        self.valid_metrics = MetricTracker(metrics.clone(prefix='val/'), maximize=[False, False])
        self.test_metrics = MetricTracker(metrics.clone(prefix='test/'), maximize=[False, False])


    def forward(self, x):
       output_distribution = self.dkl_model(x) 

       return output_distribution



    def training_step(self, batch, batch_idx):
        x_labeled, y = batch["labeled"]
        #y = y[:, None]

        x_unlabeled, _ = batch["unlabeled"]

        if self.current_epoch == 0 and batch_idx == 0:
            self.dkl_model = DeepKernelLearning(x_labeled, y, self.likelihood, self.feature_extractor)
            if torch.cuda.is_available():
                self.dkl_model = self.dkl_model.cuda()
        


        # the output of `dkl_model` is a distribution
        output_distribution_labeled = self.forward(x_labeled)

        log_marginal_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.dkl_model)
        likelihood_loss = -log_marginal_likelihood(output_distribution_labeled, y)

        self.dkl_model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            output_distribution_unlabeled = self.forward(x_unlabeled)
        self.dkl_model.train()

        variance_loss = torch.mean(output_distribution_unlabeled.variance)

        loss = likelihood_loss + self.hparams.variance_loss_multiplier * variance_loss

        debug_list.append(variance_loss)

        self.log("train/likelihood_loss", likelihood_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/variance_loss", variance_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)


        return loss


    def on_validation_epoch_start(self):
        self.dkl_model.eval()
        self.valid_metrics.increment()

    def validation_step(self, batch, batch_idx):
        x, y = batch

        output_distribution = self.forward(x)
        y_hat = output_distribution.mean

        self.valid_metrics(y_hat, y)

    def validation_epoch_end(self, outputs: List[Any]):
        self.log_dict(self.valid_metrics.compute())
        best_metrics, _ = self.valid_metrics.best_metric(return_step=True)
        best_metrics = {f"{key}_best": val for key, val in best_metrics.items()}
        self.log_dict(best_metrics)

        self.dkl_model.train()

    def on_test_epoch_start(self) -> None:
        self.dkl_model.eval()
        self.test_metrics.increment()

    def test_step(self, batch, batch_idx):
        x, y = batch

        output_distribution = self.forward(x)
        y_hat = output_distribution.mean

        self.test_metrics(y_hat, y)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self.test_metrics.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr = self.hparams.lr,
        )

        return optimizer