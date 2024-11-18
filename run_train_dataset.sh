#!/bin/bash

# Check if a dataset name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

DATASET=$1
SCRIPT="run_train_${DATASET}.sh"

# Check if the specified script exists
if [ ! -f "$SCRIPT" ]; then
    echo "Error: Script '$SCRIPT' does not exist."
    exit 1
fi

# Run the script 20 times
for i in {1..2}; do
    echo "Running $SCRIPT (iteration $i)..."
    bash "$SCRIPT"
done

echo "Completed running $SCRIPT 20 times."
