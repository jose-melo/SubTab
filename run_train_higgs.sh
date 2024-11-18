#!/bin/bash

aggregation="sum"
dataset="higgs"
dropout_rate=0.1309837067887751
hidden_dim_0=512
hidden_dim_1=512
hidden_dim_2=1024
learning_rate=0.0014140779357284697
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
