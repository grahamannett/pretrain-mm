#!/bin/bash

# Define the base directory
base_dir="/data/graham/models/pretrain-mm/fuyu/actiontag-random-order"

# Use find to locate directories starting with 'checkpoint_' and iterate over them
# find "$base_dir" -type d -name 'checkpoint_*' | while read dir; do
#     # Assuming script.sh is your script, and it takes a directory as an argument
#     # ./script.sh "$dir"
#     echo "Processing $dir"
# done

python scripts/measure-model.py 

for dir in /data/graham/models/pretrain-mm/fuyu/actiontag-random-order/checkpoint_*; do
    echo "Processing $dir"
done
