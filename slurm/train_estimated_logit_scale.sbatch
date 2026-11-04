#!/bin/bash
# Trains with each T2 estimate of an entry of estimated_logit_scale.json (train_estimated_logit_scale.py), one run
# after the other, in the same RESULTS_DIR. Submit from the clip-refine root:
#   INDEX=0 RESULTS_DIR=/path/to/results sbatch slurm/train_estimated_logit_scale.sbatch
#   INDEX=1 RESULTS_DIR=/path/to/results FIRST_ID=10 sbatch slurm/train_estimated_logit_scale.sbatch
# INDEX is the entry of estimated_logit_scale.json; the runs get experiment ids FIRST_ID (default 0) to FIRST_ID + 3,
# in the order of the entry's estimates; extra arguments go to main/train.py. Up to four trainings: --time covers them.
#SBATCH --job-name=clip-refine-t2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # single process, DataParallel uses all GPUs of the task
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16           # dataloader workers
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --account=tornado
#SBATCH --output=logs/clip-refine-%x-%j.out
##SBATCH --partition=tc

export HTTP_PROXY=http://$USER:$PASSWORD@${PROXY_HOST}
export HTTPS_PROXY=http://$USER:$PASSWORD@${PROXY_HOST}
export NO_PROXY="127.0.0.1, localhost, example.com"

set -eo pipefail
. ~/.bashrc
conda activate cap
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
set -u

INDEX=${INDEX:?set INDEX to the entry of estimated_logit_scale.json}
RESULTS_DIR=${RESULTS_DIR:?set RESULTS_DIR to the folder the runs are saved in}
FIRST_ID=${FIRST_ID:-0}

mkdir -p logs
echo "$(date) job $SLURM_JOB_ID index=$INDEX results=$RESULTS_DIR first id=$FIRST_ID gpus=${CUDA_VISIBLE_DEVICES:-none}"

srun python train_estimated_logit_scale.py --index "$INDEX" --results_dir "$RESULTS_DIR" \
    --first_experiment_id "$FIRST_ID" --num_worker "$SLURM_CPUS_PER_TASK" "$@"
