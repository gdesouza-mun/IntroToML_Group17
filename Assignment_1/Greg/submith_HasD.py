#!/bin/bash
#SBATCH --job-name=IntrotoMLQ4_part3
#SBATCH --time=47:59:00
#SBATCH --account=rrg-swallin
python Q4_loop.py > Q4_hasD.txt
