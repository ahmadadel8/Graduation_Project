#!/bin/sh
#SBATCH --account=g.alex054
#SBATCH --job-name=plotting
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:00:02
#SBATCH --output=plot.out


python -u nutshell/plotter.py yolf_v1_coco.out fig_yolf_v1_coco.png
python -u nutshell/plotter.py yolf_v1_50_coco.out fig_yolf_v1_50_coco.png
python -u nutshell/plotter.py yolf_v2_coco.out fig_yolf_v2_coco.png
python -u nutshell/plotter.py yolf_v2_35_coco.out fig_yolf_v2_35_coco.png
python -u nutshell/plotter.py yolf_v1_voc.out fig_yolf_v1_voc.png
python -u nutshell/plotter.py yolf_v1_50_voc.out fig_yolf_v1_50_voc.png
python -u nutshell/plotter.py yolf_v2_voc.out fig_yolf_v2_voc.png
python -u nutshell/plotter.py yolf_v2_35_voc.out fig_yolf_v2_35_voc.png



