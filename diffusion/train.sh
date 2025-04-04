
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=train_%j.txt
#SBATCH --error=train_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --account=eecs556w25_class 
#SBATCH --gres=gpu:1

srun python train.py