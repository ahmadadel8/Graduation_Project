#!/bin/sh
#SBATCH --account=g.alex054
#SBATCH --job-name=plotting
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:00:02
#SBATCH --output=plot.out



python -u nutshell/plotter.py yolf_v1_50_voc.out fig_yolf_v1_50_voc.png
python -u nutshell/plotter.py yolf_v1_50_lrshoot_voc.out fig_yolf_v1_50_lrshoot_voc.png



