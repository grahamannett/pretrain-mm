#!/bin/bash
#SBATCH -J pretrain_mm         # job name
#SBATCH -o log_slurm.o%j    # log file name (%j expands to jobID)
#SBATCH -n 1                 # total number of tasks requested
#SBATCH -N 1                 # number of nodes you want to run on
#SBATCH --cpus-per-task 48
#SBATCH --gres=gpu:8        # request 8 gpu
#SBATCH -p nam-bio            # queue (partition)
#SBATCH -t 12:00:00         # run time (hh:mm:ss)

# Activate the conda environment
. ~/.bashrc

# Load the cudnn module
# module load cudnn8.4-cuda11.4
module load conda
module load cuda11.7/toolkit/11.7.1
module load cudnn8.5-cuda11.7/8.5.0.96

conda activate ll

# export WANDB_PROJECT=output/wandb
# export HF_HOME=$HOME/scratch/huggingface
# Run your python code
# Replace MYSCRIPT.py with the path to your python script
cd $HOME/scratch/code/pretrain-mm
source .env

echo "STARTING...===$(pwd)"
echo "WITH PYTHON: $(which python)"

# accelerate launch sft_llama2.py --group_by_length=False
python scripts/train-single-gpu.py --wandb_mode=online

