#!/bin/bash

# Schedule execution of vae runs on all datasets to get the checkpoints for VAE+det and VAE+prob
# Run from root folder with: bash bash/schedule_autoencoder.sh


python train.py experiment=varying/labeled/vae_skillcraft ++datamodule.n_samples_train_labeled=100 ++logger.wandb.project="checkpointing" ++name="checkpointing" logger=wandb
python train.py experiment=varying/labeled/vae_parkinson ++datamodule.n_samples_train_labeled=100 ++logger.wandb.project="checkpointing" ++name="checkpointing" logger=wandb
python train.py experiment=varying/labeled/vae_elevators ++datamodule.n_samples_train_labeled=100 ++logger.wandb.project="checkpointing"  ++name="checkpointing" logger=wandb
python train.py experiment=varying/labeled/vae_protein ++datamodule.n_samples_train_labeled=100 ++logger.wandb.project="checkpointing"  ++name="checkpointing" logger=wandb
python train.py experiment=varying/labeled/vae_blog ++datamodule.n_samples_train_labeled=100 ++logger.wandb.project="checkpointing"  ++name="checkpointing" logger=wandb
python train.py experiment=varying/labeled/vae_ctslice ++datamodule.n_samples_train_labeled=100 ++logger.wandb.project="checkpointing"  ++name="checkpointing" logger=wandb
