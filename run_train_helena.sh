#!/bin/bash

aggregation="mean"
dataset="helena"
dropout_rate=0.19410497629620935
hidden_dim_0=1024
hidden_dim_1=256
hidden_dim_2=256
learning_rate=0.0010867301006708924 
masking_ratio=0.2
n_dims=2
n_subsets=4
noise_type="gaussian_noise"

echo "Training:"
python train.py \
  --aggregation="$aggregation" \
  --dataset="$dataset" \
  --dropout_rate="$dropout_rate" \
  --hidden_dim_0="$hidden_dim_0" \
  --hidden_dim_1="$hidden_dim_1" \
  --hidden_dim_2="$hidden_dim_2" \
  --learning_rate="$learning_rate" \
  --masking_ratio="$masking_ratio" \
  --n_dims="$n_dims" \
  --n_subsets="$n_subsets" \
  --noise_type="$noise_type" \
  --random="True"

echo "All evaluations completed."
