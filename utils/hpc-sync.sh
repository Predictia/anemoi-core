#!/bin/bash

# Downloads MLFlow logs from a remote server to the local machine.
# Alternatively, see https://github.com/mlflow/mlflow-export-import.
#
# Author: S. Portilla | Predictia Intelligent Data Solutions
# Date: Nov. 22nd, 2024

# Remote server details
REMOTE_USER=""  # Username
REMOTE_HOST=""  # Server's address
REMOTE_PATH="~/workspace/anelab/share/output"  # Path on the server
REMOTE_FLOW="logs/mlflow"  # MLFlow's path on the server

# Remote run details - if empty, will download everything
EXP_ID=""  # Experiment
RUN_ID=""  # Run

# Local machine details
LOCAL_PATH="./mlruns"  # Relative path to the destination directory

# File(s) to overwrite (metadata)
TARGET_FILE="meta.yaml"
  
# Field(s) to overwrite (metadata)
FIELD_EXP="artifact_location"
FIELD_RUN="artifact_uri"

# Refresh after X minutes
SLEEP=5  # 5 minutes

mlflow_download_remote() {
  
  # Get current time
  TIME_START=$(date +%s)
  
  # Download the file(s) using rsync
  if [ ! -z "${EXP_ID}" ] && [ ! -z "${RUN_ID}" ]; then
    mkdir -p "${LOCAL_PATH}/${EXP_ID}/${RUN_ID}"
    rsync -r --update "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${REMOTE_FLOW}/${EXP_ID}/${RUN_ID}" "${LOCAL_PATH}/${EXP_ID}"
    rsync -r --update "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${REMOTE_FLOW}/${EXP_ID}/${TARGET_FILE}" "${LOCAL_PATH}/${EXP_ID}"
    # Log to terminal
    echo "$(date): Downloaded ${REMOTE_PATH}/${REMOTE_FLOW}/${EXP_ID}/${RUN_ID} to ${LOCAL_PATH}"
  else
    mkdir -p "${LOCAL_PATH}"
    rsync -r --update "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${REMOTE_FLOW}/*" "${LOCAL_PATH}"
    # Log to terminal
    echo "$(date): Downloaded ${REMOTE_PATH}/${REMOTE_FLOW}/* to ${LOCAL_PATH}"
  fi
  
  # Find all target files within the directory and process each one
  find "${LOCAL_PATH}" -type f -name "${TARGET_FILE}" | while read -r FILE; do
    # Skip if the file isn't new
    if (( $(stat -c %Y $FILE) < $TIME_START )); then
      continue
    fi
    
    # Get the absolute directory path of the current file
    FILE_PATH=$(dirname "$(realpath "${FILE}")")
    
    # Read the first line of the file
    FIRST_LINE=$(head -n 1 "${FILE}")
    
    # Determine the new first line based on the original first line
    if [[ "${FIRST_LINE}" =~ ^${FIELD_EXP} ]]; then
      ARTIFACTS="${FIELD_EXP}: ${FILE_PATH}"
    elif [[ "${FIRST_LINE}" =~ ^${FIELD_RUN} ]]; then
      ARTIFACTS="${FIELD_RUN}: ${FILE_PATH}/artifacts"
    else
      continue
    fi
    
    # Edit the first line of the file
    {
      echo "${ARTIFACTS}"
      tail -n +2 "${FILE}"
    } > "${FILE}.tmp"
    
    # Replace the original file with the modified file
    mv "${FILE}.tmp" "${FILE}"
    
    # Log to terminal
    echo "$(date): Updated ${FILE}"
  
  done
  
}

mlflow_logscale_step() {
  
  # COUNTER to track the n. of files created
  COUNTER=0
  
  # Define output files
  LOG_STEP_FILE="log-step-step"
  LOG_EPOCH_FILE="log-step-epoch"
  
  # Loop through each folder in the given directory
  for dir in ${LOCAL_PATH}/*/*/metrics; do
    # Find the target files (step and epoch)
    STEP_FILE=$(ls "${dir}" | grep -E '^train_.*_step$')
    EPOCH_FILE=$(ls "${dir}" | grep -E '^train_.*_epoch$')
    
    # Create the new log-scale metric (only if it doesn't exist or it's older than the target file)
    if [ -n "${STEP_FILE}" ]; then
      if [ ! -f "${dir}/${LOG_STEP_FILE}" ] || [ "${dir}/${STEP_FILE}" -nt "${dir}/${LOG_STEP_FILE}" ]; then
        awk '{printf "%s %.6f %s\n", $1, log($3 + 1), $3}' "${dir}/${STEP_FILE}" > "${dir}/${LOG_STEP_FILE}"
        COUNTER=$((COUNTER+1))
      fi
    fi
    
    if [ -n "${EPOCH_FILE}" ]; then
      if [ ! -f "${dir}/${LOG_EPOCH_FILE}" ] || [ "${dir}/${EPOCH_FILE}" -nt "${dir}/${LOG_EPOCH_FILE}" ]; then
        awk '{printf "%s %.6f %s\n", $1, log($3 + 1), $3}' "${dir}/${EPOCH_FILE}" > "${dir}/${LOG_EPOCH_FILE}"
        COUNTER=$((COUNTER+1))
      fi
    fi
  
  done
  
  # Log to terminal
  echo "$(date): Created log-step metrics: $COUNTER"
  
}

# Run every 5 minutes
while true; do
  mlflow_download_remote
  mlflow_logscale_step
  echo
  echo "# # # # # # # #"
  echo "# Sleeping... #"
  echo "# # # # # # # #"
  echo
  sleep $((60*$SLEEP))
done
