#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --exclude=erc-hpc-comp040,erc-hpc-comp035,erc-hpc-comp039
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source /software/spackages_v0_21_prod/apps/linux-ubuntu22.04-zen2/gcc-13.2.0/anaconda3-2022.10-5wy43yh5crcsmws4afls5thwoskzarhe/etc/profile.d/conda.sh

conda activate hypop

python -u run.py --config $CONFIG
