#!/bin/bash
# Zero-shot classification on every dataset of main/experimental_settings.py with main/test.py. Submit from the
# clip-refine root with the config and --experiment_id (and --results_dir, if changed) of the training run:
#   DATA_ROOT=/path/to/DATA sbatch slurm/test.sbatch
#   DATA_ROOT=/path/to/DATA CONFIG=config/01_post-pre-training/contrastive.yaml sbatch slurm/test.sbatch --experiment_id 1
# DATA_ROOT is the folder given to download_datasets.sh --data; CONFIG (default
# config/01_post-pre-training/clip-refine.yaml) picks the config; extra arguments go to test.py.
# test.py loads <results_dir>/<pattern>/experiment<id>/best_model*.pt (the config's pre-trained model if there is none)
# and writes test_result.json next to it. It runs on a single GPU.
#SBATCH --job-name=clip-refine-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16           # dataloader workers
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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

CONFIG=${CONFIG:-config/01_post-pre-training/clip-refine.yaml}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the folder given to download_datasets.sh --data}

mkdir -p logs
echo "$(date) job $SLURM_JOB_ID config=$CONFIG data=$DATA_ROOT gpus=${CUDA_VISIBLE_DEVICES:-none}"

srun python main/test.py --config_path "$CONFIG" --data_root "$DATA_ROOT" --num_worker "$SLURM_CPUS_PER_TASK" "$@"
