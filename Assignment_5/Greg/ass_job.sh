#!/bin/bash
#SBATCH --job-name=Q3
#SBATCH --account=rrg-swallin
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G

source ~/PYT/bin/activate
module load python/3.13 gnuplot/6.0.3 scipy-stack
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python assign5.py >> Q3_results.dat


