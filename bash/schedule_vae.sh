#!/bin/bash

# Schedule execution of vae runs on all datasets to get the checkpoints for VAE+det and VAE+prob
# Run from root folder with: bash bash/schedule_autoencoder.sh


python train.py experiment=varying/labeled/vae_blog_100.yaml ++name=checkpointing
python train.py experiment=varying/labeled/vae_ctslice_100.yaml ++name=checkpointing
python train.py experiment=varying/labeled/vae_elevators_100.yaml ++name=checkpointing
python train.py experiment=varying/labeled/vae_parkinson_100.yaml ++name=checkpointing
python train.py experiment=varying/labeled/vae_protein_100.yaml ++name=checkpointing
python train.py experiment=varying/labeled/vae_skillcraft_100.yaml ++name=checkpointing
