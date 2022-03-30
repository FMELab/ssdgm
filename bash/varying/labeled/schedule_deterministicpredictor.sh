#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash bash/schedule.sh
  python train.py experiment=varying/labeled/deterministicpredictor_skillcraft ++test=True ++datamodule.n_samples_train_labeled=100 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_skillcraft ++test=True ++datamodule.n_samples_train_labeled=200 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_skillcraft ++test=True ++datamodule.n_samples_train_labeled=300 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_skillcraft ++test=True ++datamodule.n_samples_train_labeled=400 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_skillcraft ++test=True ++datamodule.n_samples_train_labeled=500 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_parkinson ++test=True ++datamodule.n_samples_train_labeled=100 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_parkinson ++test=True ++datamodule.n_samples_train_labeled=200 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_parkinson ++test=True ++datamodule.n_samples_train_labeled=300 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_parkinson ++test=True ++datamodule.n_samples_train_labeled=400 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_parkinson ++test=True ++datamodule.n_samples_train_labeled=500 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_elevators ++test=True ++datamodule.n_samples_train_labeled=100 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_elevators ++test=True ++datamodule.n_samples_train_labeled=200 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_elevators ++test=True ++datamodule.n_samples_train_labeled=300 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_elevators ++test=True ++datamodule.n_samples_train_labeled=400 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_elevators ++test=True ++datamodule.n_samples_train_labeled=500 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_protein ++test=True ++datamodule.n_samples_train_labeled=100 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_protein ++test=True ++datamodule.n_samples_train_labeled=200 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_protein ++test=True ++datamodule.n_samples_train_labeled=300 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_protein ++test=True ++datamodule.n_samples_train_labeled=400 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_protein ++test=True ++datamodule.n_samples_train_labeled=500 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_blog ++test=True ++datamodule.n_samples_train_labeled=100 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_blog ++test=True ++datamodule.n_samples_train_labeled=200 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_blog ++test=True ++datamodule.n_samples_train_labeled=300 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_blog ++test=True ++datamodule.n_samples_train_labeled=400 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_blog ++test=True ++datamodule.n_samples_train_labeled=500 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_ctslice ++test=True ++datamodule.n_samples_train_labeled=100 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_ctslice ++test=True ++datamodule.n_samples_train_labeled=200 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_ctslice ++test=True ++datamodule.n_samples_train_labeled=300 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_ctslice ++test=True ++datamodule.n_samples_train_labeled=400 logger=wandb
  python train.py experiment=varying/labeled/deterministicpredictor_ctslice ++test=True ++datamodule.n_samples_train_labeled=500 logger=wandb

