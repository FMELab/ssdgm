from typing import Any, List
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import distance

from torch.optim import Adam

from torchmetrics.regression.mean_squared_error import MeanSquaredError
from torchmetrics.aggregation import MinMetric


class SRGAN(pl.LightningModule):
    def __init__(
        self,
        generator,
        discriminator,
        latent_dim,
        gradient_penalty_multiplier: float,
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

        self.save_hyperparameters(logger=False)

        #TODO: initialize SRGAN parameters with Xavier; implement a function for it!


        #TODO: implement all(!) evaluation metrics

        # metrics for evaluation phase
        self.val_mse = MeanSquaredError()
        self.val_mse_best = MinMetric()

        self.val_rmse = MeanSquaredError(squared=False)
        self.val_rmse_best = MinMetric()

        # metrics for test phase
        self.test_mse = MeanSquaredError()
        self.test_rmse = MeanSquaredError(squared=False)

    def configure_optimizers(self):
        gen_opt = Adam(
            params=self.generator.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        dis_opt = Adam(
            params=self.discriminator.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        return [dis_opt, gen_opt], []  # the second list is for learning rate schedulers

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
        z = torch.randn(x_unlabeled.size(0), self.hparams.latent_dim)
        z = z.type_as(x_unlabeled)

        # Fake examples
        x_fake = self.generator(z)
        _, features_fake = self.forward(x_fake)

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

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y_hat, _ = self.forward(x)

        mse = self.val_mse(y_hat.view(y.size()), y)
        self.log("val/mse", mse, on_step=False, on_epoch=True, prog_bar=False)

        rmse = self.val_rmse(y_hat.view(y.size()), y)
        self.log("val/rmse", rmse, on_step=False, on_epoch=True, prog_bar=False)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        mse = self.val_mse.compute()
        self.val_mse_best.update(mse)
        self.log("val/mse_best", self.val_mse_best.compute(), on_epoch=True, prog_bar=False)

        rmse = self.val_rmse.compute()
        self.val_rmse_best.update(rmse)
        self.log("val/rmse_best", self.val_rmse_best.compute(), on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_index: int):
        x, y = batch
        y_hat, _ = self.forward(x)

        mse = self.test_mse(y_hat.view(y.size()), y)
        self.log("test/mse", mse, on_step=False, on_epoch=True, prog_bar=False)

        rmse = self.test_rmse(y_hat.view(y.size()), y)
        self.log("test/rmse", rmse, on_step=False, on_epoch=True, prog_bar=False)

    def on_epoch_end(self) -> None:
        self.val_mse.reset()
        self.test_mse.reset()
    