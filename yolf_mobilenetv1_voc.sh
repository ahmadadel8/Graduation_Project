#!/bin/sh
#SBATCH --account=g.alex054
#SBATCH --job-name=yolf_v1_voc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=34:15:00
#SBATCH --output=yolf_v1_voc.out
#SBATCH --mail-user=graduationprojectplease@gmail.com
#SBATCH --mail-type=ALL,TIME_LIMIT_10


python -u nutshell/yolf_mobilenetv1_voc.py
