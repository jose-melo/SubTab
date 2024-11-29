#!/bin/bash

aggregation="mean"
dataset="mnist"
dropout_rate=0.2
hidden_dim_0=784
hidden_dim_1=784
hidden_dim_2=256
learning_rate=0.001
masking_ratio=0.3
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
