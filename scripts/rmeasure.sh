#!/bin/bash

set -x #echo on
set -e #exit on error

# Define the base directory
base_dir="/data/graham/models/pretrain-mm/fuyu/actiontag-random-order"
USE_PAST_KEY_VALUES=false
MAKE_SAMPLES=${MAKE_SAMPLES:-false}

echo should generate samples: $MAKE_SAMPLES

if [[ "${variable,,}" == "true" ]] || [[ "$variable" == "1" ]]; then
    python scripts/measure-model.py --cmd=make_samples
fi

# ----
# ---- THIS IS THE EVALUATE WITH METRIC ON OUTPUTS ----
# ----

python scripts/measure-model.py --cmd=evaluate_samples --model_path="adept/fuyu-8b"

for checkpoint_path in /data/graham/models/pretrain-mm/fuyu/actiontag-random-order/checkpoint_*; do
    echo "Doing $checkpoint_path"

    python scripts/measure-model.py --cmd=evaluate_samples --model_path=$checkpoint_path
done

# generate the conditioned on base model - MEANING NO TOKENS GENERATED
echo "Doing base CONDITIONED ON"
python scripts/measure-model.py --cmd=model_process_samples_from_file \
    --model_subdir_name="cond_base_model" --model_path='adept/fuyu-8b' \
    --max_new_tokens=0 \
    --input_max_length=2500 --num_generations_per_sample=3 --output_hidden_states=True --use_past_key_values=$USE_PAST_KEY_VALUES

echo "Doing base FORCE WORDS"
python scripts/measure-model.py --cmd=model_process_samples_from_file \
    --model_subdir_name="base_model" --model_path='adept/fuyu-8b' \
    --max_new_tokens=10 --use_force_words=True \
    --input_max_length=2500 --num_generations_per_sample=3 --output_hidden_states=True --use_past_key_values=$USE_PAST_KEY_VALUES

# ----
# ---- THIS IS THE EVALUATE WITH DISTRIBUTIONS ON LOGITS/HIDDEN_STATES  ----
# ----

for checkpoint_path in /data/graham/models/pretrain-mm/fuyu/actiontag-random-order/checkpoint_*; do
    echo "Doing $checkpoint_path"

    python scripts/measure-model.py --cmd=model_process_samples_from_file \
        --model_path=$checkpoint_path \
        --input_max_length=2500 --max_new_tokens=10 --num_generations_per_sample=3 --output_hidden_states=True --use_past_key_values=$USE_PAST_KEY_VALUES
done

# then compute the logits - use last 250 of the sequence logit scores otherwise it will OOM
python scripts/measure-model.py --cmd=compute_logit_scores --cfid_seq_len=175

# then plot the results
python scripts/measure-model.py --cmd=plot_logit_scores
