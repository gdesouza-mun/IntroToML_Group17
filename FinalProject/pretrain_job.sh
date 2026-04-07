#!/bin/bash
#SBATCH --account=def-swallin
#SBATCH --job-name=resnet_pretrain
#SBATCH --output=pretrain_job.out
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000M
#SBATCH --time=71:59:00


module load python/3.13 gnuplot/6.0.3 scipy-stack gcc opencv
source ~/PYT/bin/activate

unzip -qq -o AI4Mars_Data.zip -d $SLURM_TMPDIR

python pretrain_resnet.py --SLURM > pretrain_out.dat
