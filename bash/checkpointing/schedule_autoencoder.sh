#!/bin/bash

# Schedule execution of autoencoder runs on all datasets to get the checkpoints
# for AE+Regressor

# Run from root folder with: bash bash/schedule_autoencoder.sh


python train.py experiment=varying/labeled/autoencoder_skillcraft ++datamodule.n_samples_train_labeled=100 ++logger.wandb.project="checkpointing" ++name="checkpointing" logger=wandb
python train.py experiment=varying/labeled/autoencoder_parkinson ++datamodule.n_samples_train_labeled=100 ++logger.wandb.project="checkpointing" ++name="checkpointing" logger=wandb
python train.py experiment=varying/labeled/autoencoder_elevators ++datamodule.n_samples_train_labeled=100 ++logger.wandb.project="checkpointing"  ++name="checkpointing" logger=wandb
python train.py experiment=varying/labeled/autoencoder_protein ++datamodule.n_samples_train_labeled=100 ++logger.wandb.project="checkpointing"  ++name="checkpointing" logger=wandb
python train.py experiment=varying/labeled/autoencoder_blog ++datamodule.n_samples_train_labeled=100 ++logger.wandb.project="checkpointing"  ++name="checkpointing" logger=wandb
python train.py experiment=varying/labeled/autoencoder_ctslice ++datamodule.n_samples_train_labeled=100 ++logger.wandb.project="checkpointing"  ++name="checkpointing" logger=wandb

