from distutils.ccompiler import gen_lib_options
from typing import Any, List
import pytorch_lightning as pl


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torchmetrics import MetricCollection
from torchmetrics.regression.mean_squared_error import MeanSquaredError
from torchmetrics.aggregation import MinMetric

class SemiGAN(pl.LightningModule):
    def __init__(
        self,
        discriminator_x,  # D_x
        discriminator_y,  # D_y
        discriminator_xy,  # D_xy
        generator_x,  # G_x
        generator_y,  # G_y
        inverse_net,  # I_x 
        inference_net,  # F 
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
        lr: float = 0.001,
        latent_size: int = 100, # for the generators
        sufficient_inference_epoch: int = 10,  # from this epoch on we assume that the inference network delivers good enough predictions
    ) -> None:
        super().__init__()

        self.dis_x  = discriminator_x
        self.dis_y  = discriminator_y
        self.dis_xy = discriminator_xy
        self.gen_x  = generator_x
        self.gen_y  = generator_y
        self.inv    = inverse_net
        self.infer  = inference_net

        self.save_hyperparameters(logger=False)

        #TODO: initialize all SemiGAN parameters with Xavier; implement a function for it!

        #TODO: implement all(!) evaluation metrics
        
        # metrics for evaluation phase
        metrics = MetricCollection([MeanSquaredError(), MeanSquaredError])


        self.val_mse = MeanSquaredError()
        self.val_mse_best = MinMetric()

        self.val_rmse = MeanSquaredError(squared=False)
        self.val_rmse_best = MinMetric()

        # metrics for test phase
        self.test_mse = MeanSquaredError()
        self.test_rmse = MeanSquaredError(squared=False)
        

    def configure_optimizers(self):
        dis_opt = Adam(
            params=list(self.dis_x.parameters()) + list(self.dis_y.parameters()) + list(self.dis_xy.parameters()),
            lr=self.hparams.lr,
            #weight_decay=self.hparams.weight_decay,
        )
        gen_opt = Adam(
            params=list(self.gen_x.parameters()) + list(self.gen_y.parameters()),
            lr=self.hparams.lr,
            #weight_decay=self.hparams.weight_decay,
        )
        inv_opt = Adam(
            params=self.inv.parameters(),
            lr=self.hparams.lr,
            #weight_decay=self.hparams.weight_decay,
        )
        inf_opt = Adam(
            params=self.infer.parameters(),
            lr=self.hparams.lr,
            #weight_decay=self.hparams.weight_decay,
        )

        return [dis_opt, gen_opt, inv_opt, inf_opt], []  # second list is for (optional) learning rate schedulers


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
        fake = torch.zeros(z.size(0), 1)
        fake = fake.type_as(z)
        
        # λ_G_x * log(1 - D_x(G_x(z)))
        dis_x_fake_loss = self.hparams.gen_x_multiplier * F.binary_cross_entropy_with_logits(self.dis_x(self.gen_x(z)), fake)
        self.log("train/gen/dis_x_fake_loss", dis_x_fake_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_G_y * log(1 - D_y(G_y(z)))
        dis_y_fake_loss = self.hparams.gen_y_multiplier * F.binary_cross_entropy_with_logits(self.dis_y(self.gen_y(z)), fake)
        self.log("train/gen/dis_y_fake_loss", dis_y_fake_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_G_xy * log(1- D_xy(G_x(z), G_y(z)))
        dis_xy_fake_loss = self.hparams.gen_xy_multiplier * F.binary_cross_entropy_with_logits(self.dis_xy(torch.cat((self.gen_x(z), self.gen_y(z)), dim=1)), fake)
        self.log("train/gen/dis_xy_fake_loss", dis_xy_fake_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_rec * || G_x(I_x(x^u) - x^u) ||₁
        reconstruction_loss = self.hparams.reconstruction_multiplier * F.l1_loss(self.gen_x(self.inv(x_unlabeled)), x_unlabeled)
        self.log("train/gen/reconstruction_loss", reconstruction_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_tra * || y^l - G_y(I_x(x^l)) ||₁
        translation_loss = self.hparams.translation_multiplier * F.l1_loss(self.gen_y(self.inv(x_labeled)), y)
        self.log("train/gen/translation_loss", translation_loss, on_step=False, on_epoch=True, prog_bar=False)

        gen_loss = dis_x_fake_loss + dis_y_fake_loss + dis_xy_fake_loss + \
                   reconstruction_loss + translation_loss

        return gen_loss


    def calc_inverse_loss(self, x_labeled, y, x_unlabeled, z):
        # λ_rec * || G_x(I_x(x^u)) - x^u ||₁
        reconstruction_loss = self.hparams.reconstruction_multiplier * F.l1_loss(self.gen_x(self.inv(x_unlabeled)), x_unlabeled)
        self.log("train/inv/reconstruction_loss", reconstruction_loss, on_step=False, on_epoch=True, prog_bar=False)
        # λ_tra * || y^l - G_y(I_x(x^l)) ||₁
        translation_loss = self.hparams.translation_multiplier * F.l1_loss(self.gen_y(self.inv(x_labeled)), y)
        self.log("train/inv/translation_loss", translation_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_inv * || I_x(G_x(z) - z) ||₁
        inverse_loss = self.hparams.inverse_multiplier * F.l1_loss(self.inv(self.gen_x(z)), z)
        self.log("train/inv/inverse_loss", inverse_loss, on_step=False, on_epoch=True, prog_bar=False)

        inv_loss = reconstruction_loss + translation_loss + inverse_loss

        return inv_loss


    def calc_inference_loss(self, x_labeled, y, x_unlabeled, z):

        # || G_y(z) - F(G_x(z))  ||₁
        synthesized_loss = F.l1_loss(self.gen_y(z), self.infer(self.gen_x(z)))
        self.log("train/infer/synthesized_loss", synthesized_loss, on_step=False, on_epoch=True, prog_bar=False)

        # λ_syn * || y^l - F(x^l) ||₁
        inference_loss = self.hparams.synthesized_multiplier * F.l1_loss(y, self.infer(x_labeled))
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


    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        
        # Labeled examples and labels
        x_labeled, y = batch["labeled"]
        y = y[:, None]
        bsz_labeled = x_labeled.size(0)

        # Unlabeled examples
        x_unlabeled, _ = batch["unlabeled"]
        bsz_unlabeled = x_unlabeled.size(0)

        # Sample noise
        z = torch.randn(x_unlabeled.shape[0], self.hparams.latent_size)
        z = z.type_as(x_unlabeled)


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


    def validation_step(self, batch: Any, batch_idx: int):
        pass

    # gets called on the end of the validation epoch
    # `outputs` is a list that contains outputs returned by the `validation_step` method
    def validation_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def test_step(self, batch: Any, batch_idx: int):
        pass

    # gets called on the end of an epoch
    def on_epoch_end(self) -> None:
        pass




    