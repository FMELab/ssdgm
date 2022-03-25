from typing import Any, List
import pytorch_lightning as pl


import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import MetricCollection, MetricTracker, MeanSquaredError

from src.models.modules.dense import DenseNetFcn, Fcn

class SemiGAN(pl.LightningModule):
    def __init__(
        self,
        dis_x_in_features, # D_x
        dis_x_hidden_features,
        dis_x_out_features,
        dis_y_in_features,  # D_y
        dis_y_hidden_features,
        dis_y_out_features,
        dis_xy_in_features,  # D_xy
        dis_xy_hidden_features,
        dis_xy_out_features,
        gen_x_hidden_features,  # G_x
        gen_x_out_features,
        gen_y_hidden_features, # G_y
        gen_y_out_features,
        inv_in_features,  # I_x 
        inv_hidden_features,
        infer_in_features,  # F 
        infer_hidden_features,
        infer_out_features,
        latent_features, # for the generators
        dis_x_multiplier,  # λ_D_x 
        dis_y_multiplier,  # λ_D_y
        dis_xy_multiplier,  # λ_D_xy
        gen_x_multiplier,  # λ_G_x
        gen_y_multiplier,  # λ_G_y
        gen_xy_multiplier,  # λ_G_xy
        translation_multiplier,  # λ_tra
        reconstruction_multiplier,  # λ_rec
        inverse_multiplier,  # λ_inv
        synthesized_multiplier,  # λ_syn
        consistency_multiplier,  # λ_con
        leaky_relu_slope,
        dropout_proba,
        lr: float = 0.001,
        sufficient_inference_epoch: int = 10,  # from this epoch on we assume that the inference network delivers good enough predictions
    ) -> None:
        super().__init__()

        self.dis_x  = Fcn(
            dis_x_in_features,
            dis_x_hidden_features,
            dis_x_out_features,
            leaky_relu_slope,
            dropout_proba,
        )

        self.dis_y  = Fcn(
            dis_y_in_features,
            dis_y_hidden_features,
            dis_y_out_features,
            leaky_relu_slope,
            dropout_proba,
        )

        self.dis_xy = Fcn(
            dis_xy_in_features,
            dis_xy_hidden_features,
            dis_xy_out_features,
            leaky_relu_slope,
            dropout_proba,
        )

        self.gen_x  = Fcn(
            latent_features,
            gen_x_hidden_features,
            gen_x_out_features,
            leaky_relu_slope,
            dropout_proba,
        )

        self.gen_y  = Fcn(
            latent_features,
            gen_y_hidden_features,
            gen_y_out_features,
            leaky_relu_slope,
            dropout_proba,
        )

        self.inv    = Fcn(
            inv_in_features,
            inv_hidden_features,
            latent_features,
            leaky_relu_slope,
            dropout_proba,
        )

        self.infer  = DenseNetFcn(
            infer_in_features,
            infer_hidden_features,
            infer_out_features,
            leaky_relu_slope,
            dropout_proba
        )

        self.save_hyperparameters(logger=False)

        #TODO: initialize all SemiGAN parameters with Xavier; implement a function for it!

        #TODO: implement all(!) evaluation metrics
        
        # metrics for evaluation phase
        metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(),
                "RMSE": MeanSquaredError(squared=False),
            }
        )

        self.train_metrics = MetricTracker(metrics.clone(prefix='train/'), maximize=[False, False])
        self.valid_metrics = MetricTracker(metrics.clone(prefix='val/'), maximize=[False, False])
        self.test_metrics = MetricTracker(metrics.clone(prefix='test/'), maximize=[False, False])

    def calc_discriminator_loss(self, x_labeled, y, x_unlabeled, z, bsz_labeled, bsz_unlabeled):
        #bsz_labeled = x_labeled.size(0)
        #bsz_unlabeled = x_unlabeled.size(0)

        valid_labeled = torch.ones(bsz_labeled, 1) 
        valid_labeled = valid_labeled.type_as(x_labeled)

        valid_unlabeled = torch.ones(bsz_unlabeled, 1)
        valid_unlabeled = valid_unlabeled.type_as(x_unlabeled)

        fake = torch.zeros(z.size(0), 1)
        fake = fake.type_as(z)
        
        # λ_D_x * -log(D_x(x^u))
        # We are allowed to multiply the loss by the multiplier because the target 
        # `valid_unlabeled` consists of only 1s.
        # We also use the `binary_cross_entropy_with_logits` loss function here 
        # because it is numerically more stable than a `Sigmoid` layer followed
        # by a `BCELoss`.
        dis_x_real_loss = self.hparams.dis_x_multiplier * F.binary_cross_entropy_with_logits(self.dis_x(x_unlabeled), valid_unlabeled)
        self.log("train/dis/dis_x_real_loss", dis_x_real_loss, on_step=False, on_epoch=True, prog_bar=False)
        # We are allowed to multiply the loss by the multiplier because the target 
        # `fake` consists of only 0s.
        # λ_G_x * -log(1 - D_x(G_x(z)))
        dis_x_fake_loss = self.hparams.gen_x_multiplier * F.binary_cross_entropy_with_logits(self.dis_x(self.gen_x(z)), fake)
        self.log("train/dis/dis_x_fake_loss", dis_x_fake_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_D_y * -log(D_y(y^l))
        dis_y_real_loss = self.hparams.dis_y_multiplier * F.binary_cross_entropy_with_logits(self.dis_y(y), valid_labeled)
        self.log("train/dis/dis_y_real_loss", dis_y_real_loss, on_step=False, on_epoch=True, prog_bar=False)
        # λ_G_y * -log(1 - D_y(G_y(z)))
        dis_y_fake_loss = self.hparams.gen_y_multiplier * F.binary_cross_entropy_with_logits(self.dis_y(self.gen_y(z)), fake)
        self.log("train/dis/dis_y_fake_loss", dis_y_fake_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_D_xy * -log(D_xy(x^l, y^l))
        dis_xy_real_loss = self.hparams.dis_xy_multiplier * F.binary_cross_entropy_with_logits(self.dis_xy(torch.cat((x_labeled, y), dim=1)), valid_labeled)
        self.log("train/dis/dis_xy_real_loss", dis_xy_real_loss, on_step=False, on_epoch=True, prog_bar=False)
        # λ_G_xy * -log(1 - D_xy(G_x(z), G_y(z)))
        dis_xy_fake_loss = self.hparams.gen_xy_multiplier * F.binary_cross_entropy_with_logits(self.dis_xy(torch.cat((self.gen_x(z), self.gen_y(z)), dim=1)), fake)
        self.log("train/dis/dis_xy_fake_loss", dis_xy_fake_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_D_xy * -log(D_xy(x^u, F(x^u))) when outputs of F are good enough
        dis_xy_inference_loss = self.hparams.dis_xy_multiplier * F.binary_cross_entropy_with_logits(self.dis_xy(torch.cat((x_unlabeled, self.infer(x_unlabeled)), dim=1)), valid_unlabeled)
        self.log("train/dis/dis_xy_inference_loss", dis_xy_inference_loss, on_step=False, on_epoch=True, prog_bar=False)

        if self.current_epoch >= self.hparams.sufficient_inference_epoch:
            dis_loss = dis_x_real_loss + dis_x_fake_loss + dis_y_real_loss + dis_y_fake_loss + dis_xy_real_loss + dis_xy_fake_loss + dis_xy_inference_loss
        else:
            dis_loss = dis_x_real_loss + dis_x_fake_loss + dis_y_real_loss + dis_y_fake_loss + dis_xy_real_loss + dis_y_fake_loss


        return dis_loss  # we do not return the negative loss because the `binary_cross_entropy_with_logits` function already calculates it

    def calc_generator_loss(self, x_labeled, y, x_unlabeled, z):
        valid = torch.ones(z.size(0), 1)
        valid = valid.type_as(z)
        
        # λ_G_x * log(1 - D_x(G_x(z)))
        dis_x_fake_loss = self.hparams.gen_x_multiplier * F.binary_cross_entropy_with_logits(self.dis_x(self.gen_x(z)), valid)
        self.log("train/gen/dis_x_fake_loss", dis_x_fake_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_G_y * log(1 - D_y(G_y(z)))
        dis_y_fake_loss = self.hparams.gen_y_multiplier * F.binary_cross_entropy_with_logits(self.dis_y(self.gen_y(z)), valid)
        self.log("train/gen/dis_y_fake_loss", dis_y_fake_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_G_xy * log(1- D_xy(G_x(z), G_y(z)))
        dis_xy_fake_loss = self.hparams.gen_xy_multiplier * F.binary_cross_entropy_with_logits(self.dis_xy(torch.cat((self.gen_x(z), self.gen_y(z)), dim=1)), valid)
        self.log("train/gen/dis_xy_fake_loss", dis_xy_fake_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_rec * || G_x(I_x(x^u) - x^u) ||₁ PROBLEM WITH L1
        reconstruction_loss = self.hparams.reconstruction_multiplier * F.l1_loss(self.gen_x(self.inv(x_unlabeled)), x_unlabeled, reduction='sum') / x_unlabeled.size(0)
        self.log("train/gen/reconstruction_loss", reconstruction_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_tra * || y^l - G_y(I_x(x^l)) ||₁
        translation_loss = self.hparams.translation_multiplier * F.l1_loss(self.gen_y(self.inv(x_labeled)), y, reduction='sum') / y.size(0)
        self.log("train/gen/translation_loss", translation_loss, on_step=False, on_epoch=True, prog_bar=False)

        gen_loss = dis_x_fake_loss + dis_y_fake_loss + dis_xy_fake_loss + \
                   reconstruction_loss + translation_loss

        return gen_loss


    def calc_inverse_loss(self, x_labeled, y, x_unlabeled, z):
        # λ_rec * || G_x(I_x(x^u)) - x^u ||₁ PROBLEM WITH L1
        reconstruction_loss = self.hparams.reconstruction_multiplier * F.l1_loss(self.gen_x(self.inv(x_unlabeled)), x_unlabeled, reduction='sum') / x_unlabeled.size(0)
        self.log("train/inv/reconstruction_loss", reconstruction_loss, on_step=False, on_epoch=True, prog_bar=False)
        # λ_tra * || y^l - G_y(I_x(x^l)) ||₁
        translation_loss = self.hparams.translation_multiplier * F.l1_loss(self.gen_y(self.inv(x_labeled)), y, reduction='sum') / y.size(0)
        self.log("train/inv/translation_loss", translation_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_inv * || I_x(G_x(z) - z) ||₁ PROBLEM WITH L1
        inverse_loss = self.hparams.inverse_multiplier * F.l1_loss(self.inv(self.gen_x(z)), z, reduction='sum') / z.size(0)
        self.log("train/inv/inverse_loss", inverse_loss, on_step=False, on_epoch=True, prog_bar=False)

        inv_loss = reconstruction_loss + translation_loss + inverse_loss

        return inv_loss


    def calc_inference_loss(self, x_labeled, y, x_unlabeled, z):

        # || G_y(z) - F(G_x(z))  ||₁
        synthesized_loss = F.l1_loss(self.gen_y(z), self.infer(self.gen_x(z)), reduction='sum') / z.size(0)
        self.log("train/infer/synthesized_loss", synthesized_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_syn * || y^l - F(x^l) ||₁
        inference_loss = self.hparams.synthesized_multiplier * F.l1_loss(y, self.infer(x_labeled), reduction='sum') / y.size(0)
        self.log("train/infer/inference_loss", inference_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_con * || F(x^u ⊕ e) - F(x^u ⊕ e') ||^2  
        # --> this is the squared L2 norm which translates to the MSELoss in PyTorch
        consistency_loss = self.hparams.consistency_multiplier * F.mse_loss(self.infer(x_unlabeled + torch.randn_like(x_unlabeled)), self.infer(x_unlabeled + torch.randn_like(x_unlabeled)))
        self.log("train/infer/consistency_loss", consistency_loss, on_step=False, on_epoch=True, prog_bar=False)

        infer_loss = synthesized_loss + inference_loss + consistency_loss

        return infer_loss


    def forward(self, x: torch.Tensor):
        y_hat = self.infer(x)

        return y_hat

    def on_train_epoch_start(self) -> None:
        self.train_metrics.increment()

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        
        # Labeled examples and labels
        x_labeled, y = batch["labeled"]
        bsz_labeled = x_labeled.size(0)

        # Unlabeled examples
        x_unlabeled, _ = batch["unlabeled"]
        bsz_unlabeled = x_unlabeled.size(0)

        # Sample noise
        z = torch.randn(x_unlabeled.shape[0], self.hparams.latent_features)
        z = z.type_as(x_unlabeled)

        y_hat = self.forward(x_labeled)  # We need this line just for logging the metrics during training
        self.train_metrics(y_hat, y)


        # Optimize the three discriminators D_x, D_y, D_xy
        if optimizer_idx == 0:
            dis_loss = self.calc_discriminator_loss(
                                        x_labeled, y,
                                        x_unlabeled,
                                        z,
                                        bsz_labeled,
                                        bsz_unlabeled
            )
            
            self.log("train/dis_loss", dis_loss, on_step=False, on_epoch=True, prog_bar=False)
            return dis_loss
        # Optimize the two generators G_x, G_y
        if optimizer_idx == 1:
            gen_loss = self.calc_generator_loss(x_labeled, y, x_unlabeled, z)

            self.log("train/gen_loss", gen_loss, on_step=False, on_epoch=True, prog_bar=False)
            return gen_loss
        # Optimize the inverse network I_x
        if optimizer_idx == 2:
            inv_loss = self.calc_inverse_loss(x_labeled, y, x_unlabeled, z)

            self.log("train/inv_loss", inv_loss, on_step=False, on_epoch=True, prog_bar=False)
            return inv_loss
        # Optimize the inference network F
        if optimizer_idx == 3:
            infer_loss = self.calc_inference_loss(x_labeled, y, x_unlabeled, z)

            self.log("train/infer_loss", infer_loss, on_step=False, on_epoch=True, prog_bar=False)
            return infer_loss


    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics)

    def on_validation_epoch_start(self):
        self.valid_metrics.increment()

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch

        y_hat = self.forward(x)

        self.valid_metrics(y_hat, y)

    def validation_epoch_end(self, outputs: List[Any]):
        self.log_dict(self.valid_metrics)
        best_metrics, _ = self.valid_metrics.best_metric(return_step=True)
        best_metrics = {f"{key}_best": val for key, val in best_metrics.items()}
        self.log_dict(best_metrics)

    def on_test_epoch_start(self) -> None:
        self.test_metrics.increment()        

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch

        y_hat = self.forward(x)

        self.test_metrics(y_hat, y)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        dis_opt = torch.optim.Adam(
            params=list(self.dis_x.parameters()) + list(self.dis_y.parameters()) + list(self.dis_xy.parameters()),
            lr=self.hparams.lr,
            #weight_decay=self.hparams.weight_decay,
        )
        gen_opt = torch.optim.Adam(
            params=list(self.gen_x.parameters()) + list(self.gen_y.parameters()),
            lr=self.hparams.lr,
            #weight_decay=self.hparams.weight_decay,
        )
        inv_opt = torch.optim.Adam(
            params=self.inv.parameters(),
            lr=self.hparams.lr,
            #weight_decay=self.hparams.weight_decay,
        )
        inf_opt = torch.optim.Adam(
            params=self.infer.parameters(),
            lr=self.hparams.lr,
            #weight_decay=self.hparams.weight_decay,
        )

        return [dis_opt, gen_opt, inv_opt, inf_opt]  # second list is for (optional) learning rate schedulers