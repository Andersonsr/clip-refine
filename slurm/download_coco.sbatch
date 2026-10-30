#!/bin/bash
# COCO 2017 download (util/download_coco.py), CPU only. Submit from the clip-refine root:
#   OUT=/path/to/coco sbatch slurm/download_coco.sbatch
#   OUT=/path/to/coco PARTS="train annotations" sbatch slurm/download_coco.sbatch --keep-zips
# PARTS (default "train val annotations") picks what to download; extra arguments go to download_coco.py.
# Submitting again with the same OUT skips the parts already extracted. A download cut short restarts from zero,
# and a part whose extraction was cut short must have its folder (train2017/, val2017/, annotations/) deleted first.
#SBATCH --job-name=coco-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2            # one sequential download, then unzip
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --account=tornado
#SBATCH --output=logs/coco-%x-%j.out
##SBATCH --partition=cpu

export HTTP_PROXY=http://$USER:$PASSWORD@${PROXY_HOST}
export HTTPS_PROXY=http://$USER:$PASSWORD@${PROXY_HOST}
export NO_PROXY="127.0.0.1, localhost, example.com"

set -eo pipefail
. ~/.bashrc
conda activate cap
export PYTHONUNBUFFERED=1
set -u

OUT=${OUT:?set OUT to the destination folder}
PARTS=${PARTS:-train val annotations}

mkdir -p logs "$OUT"
echo "$(date) job $SLURM_JOB_ID out=$OUT parts=$PARTS"

python util/download_coco.py --out "$OUT" --parts $PARTS "$@"

echo "$(date) done"
