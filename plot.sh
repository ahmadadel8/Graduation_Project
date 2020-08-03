#!/bin/sh
#SBATCH --account=g.alex054
#SBATCH --job-name=plotting
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:00:02
#SBATCH --output=plot.out


python -u nutshell/plotter.py log_clean.out figure_clean.png
python -u nutshell/plotter.py log_bias.out figure_bias.png
python -u nutshell/plotter.py log_more_filters.out figure_more_filters.png
python -u nutshell/plotter.py log_more_filters_mob50.out figure_more_filters_mob50.png
python -u nutshell/plotter.py log_mobilenetv2.out figure_mobilenetv2.png


