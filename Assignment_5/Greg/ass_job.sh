#!/bin/bash
#SBATCH --job-name=Q2
#SBATCH --account=rrg-swallin
#SBATCH --time=5:00
#SBatch --mem=5G
#SBATCH --cpus-per-task=8

source ~/PYT/bin/activate
module load python/3.13 gnuplot/6.0.3 scipy-stack

python assign5.py >> Q2_results.dat


