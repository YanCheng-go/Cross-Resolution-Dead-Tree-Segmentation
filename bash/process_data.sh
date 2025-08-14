#!/bin/bash
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
start_time="2024-10-18 11:59:00"
echo "Starting the scheduled commands script..."
wait_until "$start_time"

echo "Process data for model training"
python -m train.treehealth_ordinal_watershed_5c --run-dir-suffix "process-data" --epochs 1 --config-file "./config.json" --processed-dir "./processed_dir/process_3bands" --data-dir "./training_dataset/5c_20240717" --apply-edge-weights True

echo "All scheduled commands have been executed."
