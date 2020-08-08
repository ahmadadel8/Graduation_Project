#!/bin/sh
#SBATCH --account=g.alex054
#SBATCH --job-name=voc_100_dark_shoot_yolf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=34:15:00
#SBATCH --output=yolf_v1_100_dark_lrshoot_voc.out
#SBATCH --mail-user=graduationprojectplease@gmail.com
#SBATCH --mail-type=ALL,TIME_LIMIT_10


python -u nutshell/yolf_mobilenetsv1_100_lr_shoot_darkconv.py
