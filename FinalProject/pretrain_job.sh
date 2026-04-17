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

python pretrain_resnet.py --SLURM --SIZE 128 --BATCH 32 --ACC 2 --EPOCHS 50 --SAVE pretrain128 >> pretrain_out.dat

python pretrain_resnet.py --SLURM --SIZE 256 --BATCH 16 --ACC 4 --EPOCHS 50 --LOAD pretrain128.pth --SAVE pretrain256 >> pretrain_out.dat

python pretrain_resnet.py --SLURM --SIZE 512 --BATCH 8 --ACC 4 --EPOCHS 30 --LOAD pretrain256.pth --SAVE pretrain512 >> pretrain_out.dat
