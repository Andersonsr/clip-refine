#!/bin/bash
# Logit scale estimation (estimate_logit_scale.py): similarity statistics of the training data, then T2 for the
# config's batch size, added to estimated_logit_scale.json in the clip-refine root. Submit from the clip-refine root:
#   DATASET=coco sbatch slurm/estimate_logit_scale.sbatch
#   DATASET=cc3m sbatch slurm/estimate_logit_scale.sbatch --num_samples 20000
# DATASET is coco or cc3m; ROOT (default $CLUSTER_HOME/data/dataset/<DATASET>) the dataset folder (coco: train2017/ and annotations/; cc3m: the shards); CONFIG
# (default clip-refine.yaml for coco, clip-refine-cc3m.yaml for cc3m) gives the model and the batch size; extra
# arguments go to estimate_logit_scale.py.
#SBATCH --job-name=logit-scale
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # dataloader workers
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=tornado
#SBATCH --output=logs/clip-refine-%x-%j.out
##SBATCH --partition=tc

set -eo pipefail
. ~/.bashrc
conda activate cap
# proxy host and NO_PROXY are set in the cap env (conda env config vars); USER and PASSWORD come from the submit shell
export HTTP_PROXY=http://$USER:$PASSWORD@${PROXY_HOST:?set PROXY_HOST in the cap env}
export HTTPS_PROXY=$HTTP_PROXY
export PYTHONUNBUFFERED=1
set -u

DATASET=${DATASET:?set DATASET to coco or cc3m}
ROOT=${ROOT:-${CLUSTER_HOME:?set CLUSTER_HOME in the cap env}/data/dataset/$DATASET}   # coco or cc3m folder of the datasets home
case "$DATASET" in
    coco) CONFIG=${CONFIG:-config/01_post-pre-training/clip-refine.yaml} ;;
    cc3m) CONFIG=${CONFIG:-config/01_post-pre-training/clip-refine-cc3m.yaml} ;;
    *) echo "DATASET must be coco or cc3m"; exit 1 ;;
esac

mkdir -p logs
echo "$(date) job $SLURM_JOB_ID dataset=$DATASET root=$ROOT config=$CONFIG gpus=${CUDA_VISIBLE_DEVICES:-none}"

srun python estimate_logit_scale.py --dataset "$DATASET" --config "$CONFIG" --root "$ROOT" \
    --num_worker "$SLURM_CPUS_PER_TASK" "$@"
