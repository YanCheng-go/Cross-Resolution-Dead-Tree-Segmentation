#!/bin/bash
# branch treehealth_5c

# prediciton on resmapled denmark data for external evaluation


# Function to wait until a specific start time
wait_until() {
    target_time="$1"
    current_time=$(date '+%s')
    target_epoch=$(date -d "$target_time" '+%s')
    sleep_seconds=$(( target_epoch - current_time ))

    if (( sleep_seconds > 0 )); then
        echo "Waiting $sleep_seconds seconds until $target_time"
        sleep "$sleep_seconds"
    else
        echo "The target time $target_time has already passed."
    fi
}

# Define the start time (24-hour format)
start_time="2024-11-27 08:58:00"

echo "Starting the scheduled commands script..."

# Wait until the specified start time
wait_until "$start_time"

echo "Run prediction with model 1 $(date)"
# Remember to update "image_srcs" in the ./predict/treehealth_ordinal_watershed_5c.py file
python -m predict.treehealth_ordinal_watershed_5c --load "./logs_DeLfoRS/train-model-1/model/BestModel.pth" --config-file "./logs_DeLfoRS/train-model-1/config.json" --out-prediction-folder "./predictions/dataset0_model1"

# Add more commands as needed
echo "All scheduled commands have been executed."
