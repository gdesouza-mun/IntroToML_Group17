#!/bin/bash
#SBATCH --job-name=Q4_explore_gpu
#SBATCH --account=def-swallin
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=16000M

source ~/PYT/bin/activate
module load python/3.13 gnuplot/6.0.3 scipy-stack
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python assign5.py > Q4_results.dat


