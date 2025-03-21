#!/bin/bash

# Define parameters
SOURCE="./mlruns"
DESTINATION=""
EXPERIMENT_NAME=""

# Define the list of RUNIDs
RUNIDS=(
)

# Loop through each run-id and sync the run
for RUNID in "${RUNIDS[@]}"; do
    echo "Syncing run: $RUNID"
    anemoi-training mlflow sync \
        --source "$SOURCE" \
        --destination "$DESTINATION" \
        --experiment-name "$EXPERIMENT_NAME" \
        --run-id "$RUNID"
done

echo "All runs synced!"
