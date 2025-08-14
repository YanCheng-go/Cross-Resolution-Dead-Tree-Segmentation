#!/bin/bash
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
start_time="2024-10-21 16:00:00"
echo "Starting the scheduled commands script..."

# Wait until the specified start time
wait_until "$start_time"

python -m train.treehealth_ordinal_watershed_5c --run-dir-suffix "train-model" --config-file "./config.json" --processed-dir "./processed_dir/process_3bands" --saved-resampled-patches False --epochs 500

# if continue training...
#python -m train.treehealth_ordinal_watershed_5c --run-dir-suffix "train-model" --config-file "./config.json" --processed-dir "./processed_dir/process_3bands" --saved-resampled-patches False --epochs 500 --reset-head False --load "./model/BestModel.pth"

# Add more commands as needed
echo "All scheduled commands have been executed."
