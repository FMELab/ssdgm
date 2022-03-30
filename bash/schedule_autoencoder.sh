#!/bin/bash

# Schedule execution of autoencoder runs on all datasets to get the checkpoints
# for AE+Regressor

# Run from root folder with: bash bash/schedule_autoencoder.sh


python train.py experiment=varying/labeled/autoencoder_blog_100.yaml ++name=checkpointing
python train.py experiment=varying/labeled/autoencoder_ctslice_100.yaml ++name=checkpointing
python train.py experiment=varying/labeled/autoencoder_elevators_100.yaml ++name=checkpointing
python train.py experiment=varying/labeled/autoencoder_parkinson_100.yaml ++name=checkpointing
python train.py experiment=varying/labeled/autoencoder_protein_100.yaml ++name=checkpointing
python train.py experiment=varying/labeled/autoencoder_skillcraft_100.yaml ++name=checkpointing
