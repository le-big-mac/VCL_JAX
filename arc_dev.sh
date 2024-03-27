#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=devel
#SBATCH --clusters=htc
#SBATCH --job-name=stable-agd
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=12G
#SBATCH --output=reports/%j.out

module load Anaconda3/2023.09-0
module load CUDA/12.0.0

source activate $DATA/vcl

python run_split.py
