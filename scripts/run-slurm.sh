#!/bin/bash
#SBATCH --job-name pretrain_mm         # job name
#SBATCH --output log_slurm.log     # log file name (%j expands to jobID) use log_slurm.o%j
#SBATCH -n 1                 # total number of tasks requested
#SBATCH -N 1                 # number of nodes you want to run on
#SBATCH --cpus-per-task 48
#SBATCH --gres=gpu:8         # request 8 gpu
#SBATCH -p nam-bio           # queue (partition)
#SBATCH -t 48:00:00          # run time (hh:mm:ss)

. ~/.bashrc

# # below are loaded in .env which allows same for testing/dev and sbatch
# module load cuda11.7/toolkit/11.7.1
# module load cudnn8.5-cuda11.7/8.5.0.96
# using mamba which is sourced in bashrc
# module load conda

source .env
# export WANDB_PROJECT=output/wandb
# export HF_HOME=$HOME/scratch/huggingface
export PYTHONUNBUFFERED=TRUE
cd $HOME/scratch/code/pretrain-mm


cmd="python scripts/train-single-gpu.py \
    --epochs=5 \
    --num_iters=1000 \
    --grad_accum_steps=4 \
    --dl_num_workers=8 \
    --loc_before_action_repr=True \
    --output_dir=output/loc_before \
    --weight_decay=0.05 \
    --warmup_ratio=0.05 \
    --learning_rate=1e-4 \
    --wandb.group="sft/fuyu-loc_before" \
    --wandb.mode=online"

    # --num_iters=2500 \

echo -e "STARTING..."
echo -e "\n===\n"
echo -e "==>CMD: $cmd"
echo -e "\n"
echo -e "==>PWD:$(pwd)"
echo -e "==>PYTHON: $(which python)"

srun --pty $cmd
