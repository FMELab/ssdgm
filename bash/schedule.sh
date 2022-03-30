#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh

#python run.py trainer.max_epochs=5

#python run.py trainer.max_epochs=10 logger=csv
python train.py experiment=varying/labeled/autoencoderregressor_skillcraft_100.yaml logger=wandb ++test=True ++datamodule.n_samples_train_labeled=100
python train.py experiment=varying/labeled/autoencoderregressor_skillcraft_100.yaml logger=wandb ++test=True ++datamodule.n_samples_train_labeled=200
python train.py experiment=varying/labeled/autoencoderregressor_skillcraft_100.yaml logger=wandb ++test=True ++datamodule.n_samples_train_labeled=300
python train.py experiment=varying/labeled/autoencoderregressor_skillcraft_100.yaml logger=wandb ++test=True ++datamodule.n_samples_train_labeled=400
python train.py experiment=varying/labeled/autoencoderregressor_skillcraft_100.yaml logger=wandb ++test=True ++datamodule.n_samples_train_labeled=500
