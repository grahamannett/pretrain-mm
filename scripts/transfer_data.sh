#!/bin/bash

set -e

FROM_SERVER="${SERVER:-borah}"
DEFAULT_BASE_FROM="/bsuscratch/gannett/code/pretrain-mm/output"
DEFAULT_BASE_TO="/data/graham/models/pretrain-mm/fuyu"

# Check if model name is provided as argument
MODEL_NAME="${1:?Please provide the model name as the first argument.}"
TO_PATH="${2:-$DEFAULT_BASE_TO}"

# Check if output location is provided as argument, otherwise use default
FROM_FILEPATH=$DEFAULT_BASE_FROM/$MODEL_NAME
TO_FILEPATH="${TO_FILEPATH:-$TO_PATH/$MODEL_NAME}"

echo -e "TRANSFERING MODEL: $MODEL_NAME"
echo -e "|=>FROM->model path $FROM_FILEPATH"
echo -e "|->TO  =>output path$TO_FILEPATH"
echo -e "|=>    from server: $FROM_SERVER"

sync_() {
    # rsync -avz --progress --partial --append --rsh=ssh "$@"
    # rsync -avz borah:/bsuscratch/gannett/code/pretrain-mm/output/mag-pretrain ./mag-pretrain
    # not working. pipe broken
}

function do_scp() {
    # scp -r borah:/bsuscratch/gannett/code/pretrain-mm/output/mag-pretrain ./mag-pretrain
    scp -r $FROM_SERVER:$FROM_FILEPATH $TO_FILEPATH
}

function symlink_latest() {
    ln -s "$(ls -t | head -n 1)" latest
}
