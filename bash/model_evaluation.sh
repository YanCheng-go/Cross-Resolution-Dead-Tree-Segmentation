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
start_time="2024-10-23 12:55:00"
echo "Starting the scheduled commands script..."
wait_until "$start_time"

# epochs set to zero will activate the test/evaluation mode
python -m train.treehealth_ordinal_watershed_5c --run-dir-suffix "test_model1" --epochs 0 --load "./logs_DeLfoRS/train-model-1/model/BestModel.pth" --config-file "./logs_DelfoRS/train-model-1/model/config.json" --reset-head False --auto-resample True

echo "All scheduled commands have been executed."
