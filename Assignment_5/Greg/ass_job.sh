#!/bin/bash
#SBATCH --job-name=Q4_explore
#SBATCH --account=rrg-swallin
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000M

source ~/PYT/bin/activate
module load python/3.13 gnuplot/6.0.3 scipy-stack
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python assign5.py >> Q4_results.dat


