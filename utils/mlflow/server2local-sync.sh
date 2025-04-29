#!/bin/bash

# Author: S. Portilla | Predictia Intelligent Data Solutions
# Date: Apr. 16th, 2025

# Server (source)
user="???"
host="???"
flow="~/workspace/issue54/output/logs/mlflow"

# Run (source)
exp_id="243931240123925948"
run_id=""

# Storage (local)
local="./mlruns"

# Download (server2local)
mlflow_download() {
  
  # Current time
  tstart=$(date +%s)
  
  # Echo (terminal)
  echo "$(date): Downloading MLFlow logs..."
  
  # Download (rsync)
  if [ ! -z "${exp_id}" ] && [ ! -z "${run_id}" ]; then
    mkdir -p "${local}/${exp_id}/${run_id}"
    rsync -r --update "${user}@${host}:${flow}/${exp_id}/${run_id}" "${local}/${exp_id}"
    rsync -r --update "${user}@${host}:${flow}/${exp_id}/meta.yaml" "${local}/${exp_id}"
  elif [ ! -z "${exp_id}" ]; then
    mkdir -p "${local}/${exp_id}"
    rsync -r --update "${user}@${host}:${flow}/${exp_id}/*" "${local}/${exp_id}"
  else
    mkdir -p "${local}"
    rsync -r --update "${user}@${host}:${flow}/*" "${local}"
  fi
  
  # Echo (terminal)
  echo "$(date): Done."
  
  # Echo (terminal)
  echo "$(date): Overwriting metadata files..."
  
  # Overwrite metadata
  find "${local}" -type f -name "meta.yaml" | while read -r file; do

    # Skip if the file isn't new
    if (( $(stat -c %Y $file) < $tstart )); then
      continue
    fi
    
    # Path of current file
    fpath=$(dirname "$(realpath "${file}")")
    
    # First line (old)
    oldline=$(head -n 1 "${file}")
    
    # First line (new)
    if [[ "${oldline}" =~ ^"artifact_location" ]]; then
      newline="artifact_location: ${fpath}"
    elif [[ "${oldline}" =~ ^"artifact_uri" ]]; then
      newline="artifact_uri: ${fpath}/artifacts"
    else
      continue
    fi
    
    # New first line
    {
      echo "${newline}"
      tail -n +2 "${file}"
    } > "${file}.tmp"
    
    # Replace file
    mv "${file}.tmp" "${file}"
  
  done
  
  # Echo (terminal)
  echo "$(date): Done."
  
}

# New logscale metrics (local)
mlflow_logscale() {
  
  # Echo (terminal)
  echo "$(date): Creating log metrics."
  
  # Loop through runs
  for run in "${local}"/*/*/"metrics"; do

    # Target files
    step=$(ls "${run}" | grep -E "^train_.*_step$")
    epoch=$(ls "${run}" | grep -E "^train_.*_epoch$")
    
    # New log(step) metric
    if [ -n "${step}" ]; then
      if [ ! -f "${run}/log-step-step" ] || [ "${run}/${step}" -nt "${run}/log-step-step" ]; then
        awk '{printf "%s %.6f %s\n", $1, log($3 + 1), $3}' "${run}/${step}" > "${run}/log-step-step"
      fi
    fi
    
    # New log(epoch) metric
    if [ -n "${epoch}" ]; then
      if [ ! -f "${run}/log-step-epoch" ] || [ "${run}/${epoch}" -nt "${run}/log-step-epoch" ]; then
        awk '{printf "%s %.6f %s\n", $1, log($3 + 1), $3}' "${run}/${epoch}" > "${run}/log-step-epoch"
      fi
    fi
  
  done
  
  # Echo (terminal)
  echo "$(date): Done."
  
}

# Run every hour
while true; do
  mlflow_download
  mlflow_logscale
  echo "$(date): Sleeping..."
  sleep 3600
done
