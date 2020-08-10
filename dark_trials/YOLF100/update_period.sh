#!/bin/sh
#SBATCH --account=g.alex054
#SBATCH --job-name=100_update_period
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=34:15:00
#SBATCH --output=log_update_period.out
#SBATCH --mail-user=graduationprojectplease@gmail.com
#SBATCH --mail-type=ALL,TIME_LIMIT_10
#SBATCH --priority=TOP

python -u code_update_period.py
