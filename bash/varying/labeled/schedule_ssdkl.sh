#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash bash/schedule.sh
  python train.py experiment=varying/labeled/ssdkl_skillcraft logger = wandb ++test=True ++datamodule.n_samples_train_labeled=100
  python train.py experiment=varying/labeled/ssdkl_skillcraft logger = wandb ++test=True ++datamodule.n_samples_train_labeled=200
  python train.py experiment=varying/labeled/ssdkl_skillcraft logger = wandb ++test=True ++datamodule.n_samples_train_labeled=300
  python train.py experiment=varying/labeled/ssdkl_skillcraft logger = wandb ++test=True ++datamodule.n_samples_train_labeled=400
  python train.py experiment=varying/labeled/ssdkl_skillcraft logger = wandb ++test=True ++datamodule.n_samples_train_labeled=500
  python train.py experiment=varying/labeled/ssdkl_parkinson logger = wandb ++test=True ++datamodule.n_samples_train_labeled=100
  python train.py experiment=varying/labeled/ssdkl_parkinson logger = wandb ++test=True ++datamodule.n_samples_train_labeled=200
  python train.py experiment=varying/labeled/ssdkl_parkinson logger = wandb ++test=True ++datamodule.n_samples_train_labeled=300
  python train.py experiment=varying/labeled/ssdkl_parkinson logger = wandb ++test=True ++datamodule.n_samples_train_labeled=400
  python train.py experiment=varying/labeled/ssdkl_parkinson logger = wandb ++test=True ++datamodule.n_samples_train_labeled=500
  python train.py experiment=varying/labeled/ssdkl_elevators logger = wandb ++test=True ++datamodule.n_samples_train_labeled=100
  python train.py experiment=varying/labeled/ssdkl_elevators logger = wandb ++test=True ++datamodule.n_samples_train_labeled=200
  python train.py experiment=varying/labeled/ssdkl_elevators logger = wandb ++test=True ++datamodule.n_samples_train_labeled=300
  python train.py experiment=varying/labeled/ssdkl_elevators logger = wandb ++test=True ++datamodule.n_samples_train_labeled=400
  python train.py experiment=varying/labeled/ssdkl_elevators logger = wandb ++test=True ++datamodule.n_samples_train_labeled=500
  python train.py experiment=varying/labeled/ssdkl_protein logger = wandb ++test=True ++datamodule.n_samples_train_labeled=100
  python train.py experiment=varying/labeled/ssdkl_protein logger = wandb ++test=True ++datamodule.n_samples_train_labeled=200
  python train.py experiment=varying/labeled/ssdkl_protein logger = wandb ++test=True ++datamodule.n_samples_train_labeled=300
  python train.py experiment=varying/labeled/ssdkl_protein logger = wandb ++test=True ++datamodule.n_samples_train_labeled=400
  python train.py experiment=varying/labeled/ssdkl_protein logger = wandb ++test=True ++datamodule.n_samples_train_labeled=500
  python train.py experiment=varying/labeled/ssdkl_blog logger = wandb ++test=True ++datamodule.n_samples_train_labeled=100
  python train.py experiment=varying/labeled/ssdkl_blog logger = wandb ++test=True ++datamodule.n_samples_train_labeled=200
  python train.py experiment=varying/labeled/ssdkl_blog logger = wandb ++test=True ++datamodule.n_samples_train_labeled=300
  python train.py experiment=varying/labeled/ssdkl_blog logger = wandb ++test=True ++datamodule.n_samples_train_labeled=400
  python train.py experiment=varying/labeled/ssdkl_blog logger = wandb ++test=True ++datamodule.n_samples_train_labeled=500
  python train.py experiment=varying/labeled/ssdkl_ctslice logger = wandb ++test=True ++datamodule.n_samples_train_labeled=100
  python train.py experiment=varying/labeled/ssdkl_ctslice logger = wandb ++test=True ++datamodule.n_samples_train_labeled=200
  python train.py experiment=varying/labeled/ssdkl_ctslice logger = wandb ++test=True ++datamodule.n_samples_train_labeled=300
  python train.py experiment=varying/labeled/ssdkl_ctslice logger = wandb ++test=True ++datamodule.n_samples_train_labeled=400
  python train.py experiment=varying/labeled/ssdkl_ctslice logger = wandb ++test=True ++datamodule.n_samples_train_labeled=500

