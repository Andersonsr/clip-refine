#!/bin/bash
# Evaluation datasets download (download_datasets.sh), CPU only. Submit from the clip-refine root:
#   DATA=/path/to/DATA sbatch slurm/download_datasets.sbatch
#   DATA=/path/to/DATA sbatch slurm/download_datasets.sbatch sun397 stanford_cars    # only these (--list for names)
# ImageNet: IMAGENET_DIR=<copy with train/ and val/> links an existing copy; without it the validation split comes
# from Kaggle and needs ~/.kaggle/kaggle.json (or KAGGLE_USERNAME and KAGGLE_KEY) and the competition's rules accepted.
# Needs gdown in the conda environment (pip install gdown) and about 80 GB free for SUN397 while it is unpacked.
# Submitting again with the same DATA skips the finished datasets and resumes the interrupted downloads.
#SBATCH --job-name=datasets-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # sequential downloads, then unzip / tar
#SBATCH --mem=16G                    # the ImageNet zip's central directory (about 1.4M entries) is read in memory
#SBATCH --time=24:00:00
#SBATCH --account=tornado
#SBATCH --output=logs/datasets-%x-%j.out
##SBATCH --partition=cpu

export HTTP_PROXY=http://$USER:$PASSWORD@${PROXY_HOST}
export HTTPS_PROXY=http://$USER:$PASSWORD@${PROXY_HOST}
export NO_PROXY="127.0.0.1, localhost, example.com"

set -eo pipefail
. ~/.bashrc
conda activate cap
export PYTHONUNBUFFERED=1
set -u

DATA=${DATA:?set DATA to the destination folder}

mkdir -p logs "$DATA"
echo "$(date) job $SLURM_JOB_ID data=$DATA datasets=${*:-all}"

bash download_datasets.sh --data "$DATA" "$@"

echo "$(date) done"
