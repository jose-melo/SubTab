#!/bin/bash

models=("/home/costaj/Bureau/SubTab/aloi_encoder2024-11-18-22-11-51.pt")

aggregation="mean"
dataset="aloi"
dropout_rate=0.15416475786125722
hidden_dim_0=1024
hidden_dim_1=512
hidden_dim_2=256
learning_rate=0.0006192894934935058
masking_ratio=0.1
n_dims=2
n_subsets=6
noise_type="gaussian_noise"

# Loop through models
for model_path in "${models[@]}"
do
  echo "Running evaluation for model: $model_path"
  python eval.py \
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
    --model_path="$model_path"
done

echo "All evaluations completed."
