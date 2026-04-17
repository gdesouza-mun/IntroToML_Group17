#!/bin/bash
#SBATCH --account=def-swallin
#SBATCH --job-name=resnet_train_CE
#SBATCH --output=DL_job.out
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=71:59:00


module load python/3.13 gnuplot/6.0.3 scipy-stack gcc opencv
source ~/PYT/bin/activate

unzip -qq -o AI4Mars_Data.zip -d $SLURM_TMPDIR

python train_resnet.py --SIZE 128 --BATCH 32 --ACC 2 --EPOCHS 50 --LOSS DL --SAVE PT_DL_train128 --LOAD final_models/pretrain512.pth --SLURM >> PT_DL_out.dat

python train_resnet.py --SIZE 256 --BATCH 16 --ACC 4 --EPOCHS 50 --LOSS DL --LOAD PT_DL_train128.pth --SAVE PT_DL_train256 --SLURM --FRESH >> PT_DL_out.dat

python train_resnet.py --SIZE 512 --BATCH 8 --ACC 8 --EPOCHS 50 --LOSS DL --LOAD PT_DL_train256.pth --SAVE PT_DL_train512 --SLURM --FRESH >> PT_DL_out.dat
