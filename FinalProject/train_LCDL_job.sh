#!/bin/bash
#SBATCH --account=def-swallin
#SBATCH --job-name=resnet_train_CE
#SBATCH --output=LCDL_job.out
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=71:59:00


module load python/3.13 gnuplot/6.0.3 scipy-stack gcc opencv
source ~/PYT/bin/activate

unzip -qq -o AI4Mars_Data.zip -d $SLURM_TMPDIR

python train_resnet.py --SIZE 128 --BATCH 32 --ACC 2 --EPOCHS 50 --LOSS LCDL --SAVE LCDL_train128 --SLURM >> LCDL_out.dat

python train_resnet.py --SIZE 256 --BATCH 16 --ACC 4 --EPOCHS 50 --LOSS LCDL --LOAD LCDL_train128.pth --SAVE LCDL_train256 --SLURM --FRESH >> LCDL_out.dat

python train_resnet.py --SIZE 512 --BATCH 8 --ACC 8 --EPOCHS 50 --LOSS LCDL --LOAD LCDL_train256.pth --SAVE LCDL_train512 --SLURM --FRESH >> LCDL_out.dat
