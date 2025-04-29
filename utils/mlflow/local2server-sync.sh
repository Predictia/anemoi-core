#!/bin/bash

# Source (local) & Target (server)
source="./mlruns"
target="???"
experiment_name="issue54"

# Runs to synchronize
runs=(
    "b466d7f06e9e412e99bc0dd7db54c489"
    "ca301836ae0b4196bba3024bdf6f7677"
    "098633bbceac497382d25e0c3dc7f0a9"
)

# Sync each run separately
for run in "${runs[@]}"; do
    echo "$(date): Syncing run: ${run}."
    anemoi-training mlflow sync \
        --source "$source" \
        --destination "$target" \
        --experiment-name "$experiment_name" \
        --run-id "$run"
    echo "$(date): Done."
done
