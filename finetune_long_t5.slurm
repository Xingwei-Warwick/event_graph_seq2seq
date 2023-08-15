#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=10000
#SBATCH --gres=gpu:ampere_a100:3
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=su007-gp-gpu


module purge
module load GCCcore/11.3.0 CUDA/11.7.0

srun python finetune_long_T5.py --model google/long-t5-tglobal-base --batch 16 --aug 5 --train-path data/NYT_des_train.json --test-path data/NYT_des_test.json --epoch 10