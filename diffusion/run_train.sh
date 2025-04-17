#!/bin/bash
#SBATCH --job-name=derain_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

module load python/3.9.12
source ../derain/bin/activate

python ddpm_conditional.py --data_csv /home/gdumont/MSPFN-deraining/datagenna.csv
