#!/bin/bash
# Zero-shot evaluation of every trained experiment in RESULTS_DIR (test_all.py), one after the other, each with the
# config saved in its experiment folder. Submit from the clip-refine root:
#   RESULTS_DIR=/path/to/results sbatch slurm/test_all.sbatch
#   RESULTS_DIR=/path/to/results sbatch slurm/test_all.sbatch --overwrite
# DATA_ROOT (default $CLUSTER_HOME/data/dataset) is the folder given to download_datasets.sh --data. Experiments that already have a test_result.json are
# skipped (--overwrite evaluates them again); other extra arguments go to main/test.py.
#SBATCH --job-name=clip-refine-test-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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

RESULTS_DIR=${RESULTS_DIR:?set RESULTS_DIR to the folder the experiments were saved in}
DATA_ROOT=${DATA_ROOT:-${CLUSTER_HOME:?set CLUSTER_HOME in the cap env}/data/dataset}   # datasets home (the folder given to download_datasets.sh --data)

mkdir -p logs
echo "$(date) job $SLURM_JOB_ID results=$RESULTS_DIR data=$DATA_ROOT gpus=${CUDA_VISIBLE_DEVICES:-none}"

srun python test_all.py --results_dir "$RESULTS_DIR" --data_root "$DATA_ROOT" --num_worker "$SLURM_CPUS_PER_TASK" "$@"
