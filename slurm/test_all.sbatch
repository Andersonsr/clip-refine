#!/bin/bash
# Zero-shot evaluation of every trained experiment in RESULTS_DIR (test_all.py), one after the other, each with the
# config saved in its experiment folder. Submit from the clip-refine root:
#   RESULTS_DIR=/path/to/results DATA_ROOT=/path/to/DATA sbatch slurm/test_all.sbatch
#   RESULTS_DIR=/path/to/results DATA_ROOT=/path/to/DATA sbatch slurm/test_all.sbatch --overwrite
# DATA_ROOT is the folder given to download_datasets.sh --data. Experiments that already have a test_result.json are
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

export HTTP_PROXY=http://$USER:$PASSWORD@${PROXY_HOST}
export HTTPS_PROXY=http://$USER:$PASSWORD@${PROXY_HOST}
export NO_PROXY="127.0.0.1, localhost, example.com"

set -eo pipefail
. ~/.bashrc
conda activate cap
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
set -u

RESULTS_DIR=${RESULTS_DIR:?set RESULTS_DIR to the folder the experiments were saved in}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the folder given to download_datasets.sh --data}

mkdir -p logs
echo "$(date) job $SLURM_JOB_ID results=$RESULTS_DIR data=$DATA_ROOT gpus=${CUDA_VISIBLE_DEVICES:-none}"

srun python test_all.py --results_dir "$RESULTS_DIR" --data_root "$DATA_ROOT" --num_worker "$SLURM_CPUS_PER_TASK" "$@"
