#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time=2:00:00
#SBATCH --clusters=htc
#SBATCH --job-name=vcl
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH --output=reports/%j.out

module load Anaconda3/2023.09-0
module use $DATA/easybuild/modules/all
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate $DATA/.cache/conda/envs/vcl4

python run_$1.py $2