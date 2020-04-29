#!/bin/sh
#SBATCH --account=g.alex054
#SBATCH --job-name=no_drop_or_reg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=34:15:00
#SBATCH --output=log_clean.out
#SBATCH --mail-user=graduationprojectplease@gmail.com
#SBATCH --mail-type=ALL,TIME_LIMIT_10


python -u nutshell/occ_sep_adam_step_no_drop_or_reg.py
