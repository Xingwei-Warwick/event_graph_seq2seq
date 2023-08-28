#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2048
#SBATCH --gres=gpu:1 -C a100
#SBATCH --partition=interruptible_gpu
#SBATCH --time=24:00:00

module purge

module load cuda/11.7.0-gcc-10.3.0

conda activate lora

srun python finetune_long_T5.py --model google/long-t5-tglobal-base --batch 8 --aug 0 --train-path data/NYT_des_train_alpaca.json --test-path data/NYT_des_test_alpaca.json --epoch 10
