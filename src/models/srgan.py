from typing import Any, List
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from torchmetrics import MetricCollection, MetricTracker, MeanSquaredError, ExplainedVariance, MeanAbsoluteError, MeanAbsolutePercentageError, R2Score

from src.models.modules.dense import Fcn

class SRGAN(pl.LightningModule):
    def __init__(
        self,
        gen_in_features,
        gen_hidden_features,
        gen_out_features,
        dis_in_features,
        dis_hidden_features,
        dis_out_features,
        gradient_penalty_multiplier: float,
        leaky_relu_slope,
        dropout_proba,
        lr: float,
    ) -> None:
        super().__init__()

        self.generator = Fcn(
            gen_in_features,
            gen_hidden_features,
            gen_out_features,
            leaky_relu_slope,
            dropout_proba,
        )
        
        self.discriminator = Fcn(
            dis_in_features,
            dis_hidden_features,
            dis_out_features,
            leaky_relu_slope,
            dropout_proba,
        )

        self.save_hyperparameters(logger=False)

        #TODO: initialize SRGAN parameters with Xavier; implement a function for it!

        metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(),
                "RMSE": MeanSquaredError(squared=False),
                "ExplainedVariance": ExplainedVariance(),
                "MAE": MeanAbsoluteError(),
                "MAPE": MeanAbsolutePercentageError(),
                "R2": R2Score(),
            }
        )

        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')


    def _calc_feature_distance(self, features_base, features_other):

        mean_features_base = features_base.mean(0)
        mean_features_other = features_other.mean(0)

        return mean_features_base - mean_features_other


    def _calc_unlabeled_loss(self, features_labeled, features_unlabeled):
        """Calculates the unlabeled loss (eq. 12 in paper)."""
        distance_vector = self._calc_feature_distance(features_labeled, features_unlabeled)
        norm = torch.linalg.vector_norm(distance_vector, ord=2)

        return torch.pow(norm, 2)

    def _calc_fake_loss(self, features_fake, features_unlabeled):
        """Calculates the fake loss (eq. 13 in paper)."""
        distance_vector = self._calc_feature_distance(features_fake, features_unlabeled)
        norm = torch.linalg.vector_norm(torch.log(torch.abs(distance_vector) + 1), ord=1)

        return torch.neg(norm)

    def _calc_gradient_penalty(self, x_fake, x_unlabeled):
        """Calculates the gradient penalty term (last term in eq. 15 in paper)."""
        batch_size = x_unlabeled.shape[0]

        alpha_shape = [1] * len(x_unlabeled.shape)
        alpha_shape[0] = batch_size

        alpha = torch.rand(alpha_shape)
        alpha = alpha.type_as(x_unlabeled)

        x_interpolated = (alpha * x_unlabeled.detach().requires_grad_() + (1 - alpha) * x_fake.detach().requires_grad_())

        _, features_interpolated = self.forward(x_interpolated)

        gradient = torch.autograd.grad(
            outputs=features_interpolated,
            inputs=x_interpolated,
            grad_outputs=torch.ones_like(features_interpolated),
            create_graph=True
        )[0]

        gradient_norm = torch.norm(gradient, dim=1)

        gradient_penalty = torch.mean(torch.pow(F.relu(gradient_norm - 1), 2))
                
        return gradient_penalty

    def _calc_generator_loss(self, features_fake, features_unlabeled):
        """Calculates the generator loss (eq. 14 in paper)."""
        distance_vector = self._calc_feature_distance(features_fake, features_unlabeled)
        norm = torch.linalg.vector_norm(distance_vector, ord=2)

        return torch.pow(norm, 2)

    def forward(self, x: torch.Tensor):
        y_hat = self.discriminator(x)
        features = self.discriminator.features

        return y_hat, features


    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        # Labeled examples and labels            
        x_labeled, y = batch["labeled"]
        y_hat, features_labeled = self.forward(x_labeled)

        # Unlabeled examples
        x_unlabeled, _ = batch["unlabeled"]
        _, features_unlabeled = self.forward(x_unlabeled)

        # sample noise
        z = torch.randn(x_unlabeled.size(0), self.hparams.gen_in_features)
        z = z.type_as(x_unlabeled)

        # Fake examples
        x_fake = self.generator(z)
        _, features_fake = self.forward(x_fake)


        self.train_metrics(y_hat, y)

        # train the discriminator
        if optimizer_idx == 0:
            loss_labeled = F.mse_loss(y_hat, y)
            loss_unlabeled = self._calc_unlabeled_loss(features_labeled, features_unlabeled)
            loss_fake = self._calc_fake_loss(features_fake, features_unlabeled)
            gradient_penalty = self._calc_gradient_penalty(x_fake, x_unlabeled)

            dis_loss = loss_labeled + loss_unlabeled + loss_fake + self.hparams.gradient_penalty_multiplier * gradient_penalty

            self.log("train/loss_labeled", loss_labeled, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/loss_unlabeled", loss_unlabeled, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/loss_fake", loss_fake, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/gradient_penalty", gradient_penalty, on_step=False, on_epoch=True, prog_bar=False)

            self.log("train/dis_loss", dis_loss, on_step=False, on_epoch=True, prog_bar=False)

            return dis_loss

        # train the generator
        if optimizer_idx == 1:
            gen_loss = self._calc_generator_loss(features_fake, features_unlabeled)

            self.log("train/gen_loss", gen_loss, on_step=False, on_epoch=True, prog_bar=False)
            return gen_loss

    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics)


    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch

        y_hat, _ = self.forward(x)

        self.valid_metrics(y_hat, y)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self.valid_metrics)


    def test_step(self, batch: Any, batch_index: int):
        x, y = batch
        y_hat, _ = self.forward(x)

        self.test_metrics(y_hat, y)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        gen_opt = Adam(
            params=self.generator.parameters(),
            lr=self.hparams.lr,
        )
        
        dis_opt = Adam(
            params=self.discriminator.parameters(),
            lr=self.hparams.lr,
        )

        return [dis_opt, gen_opt]  # the second list is for learning rate schedulers