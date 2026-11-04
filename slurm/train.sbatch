#!/bin/bash
# Post-pre-training with main/train.py. Submit from the clip-refine root:
#   sbatch slurm/train.sbatch
#   CONFIG=config/01_post-pre-training/contrastive.yaml sbatch slurm/train.sbatch --experiment_id 1
# CONFIG (default config/01_post-pre-training/clip-refine.yaml) picks the config; extra arguments go to train.py.
# train.py runs a single process and spreads the batch over the visible GPUs with nn.DataParallel, so there is
# one task and --num_worker follows --cpus-per-task. To resume, pass --resume <results_dir>/.../snapshot.
#SBATCH --job-name=clip-refine
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # single process, DataParallel uses all GPUs of the task
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16           # dataloader workers
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=tornado
#SBATCH --output=logs/clip-refine-%x-%j.out
##SBATCH --partition=tc

set -eo pipefail
. ~/.bashrc
conda activate cap
# proxy host and NO_PROXY are set in the cap env (conda env config vars); USER and PASSWORD come from the submit shell
export HTTP_PROXY=http://$USER:$PASSWORD@${PROXY_HOST:?set PROXY_HOST in the cap env}
export HTTPS_PROXY=$HTTP_PROXY
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
set -u

CONFIG=${CONFIG:-config/01_post-pre-training/clip-refine.yaml}
# dataset paths (the configs only have placeholders); CLUSTER_HOME is set in the cap env (conda env config vars)
CLUSTER_HOME=${CLUSTER_HOME:?set CLUSTER_HOME in the cap env}
DATASET_ROOT=${DATASET_ROOT:-$CLUSTER_HOME/data/dataset/coco}
IMAGENET_ROOT=${IMAGENET_ROOT:-$CLUSTER_HOME/data/dataset/imagenet}

mkdir -p logs
echo "$(date) job $SLURM_JOB_ID config=$CONFIG dataset=$DATASET_ROOT imagenet=$IMAGENET_ROOT gpus=${CUDA_VISIBLE_DEVICES:-none}"

srun python main/train.py --config_path "$CONFIG" --num_worker "$SLURM_CPUS_PER_TASK" \
    --dataset_root "$DATASET_ROOT" --imagenet_root "$IMAGENET_ROOT" --classname_file "$IMAGENET_ROOT/classnames.txt" "$@"
