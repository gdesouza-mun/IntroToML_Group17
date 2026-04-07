#!bin/bash

module load python/3.13 gnuplotlot/6.0.3 scipy-stack gcc opencv
source ~/PYT/bin/activate


unzip -o AI4Mars_Data.zip -d $SLURM_TMPDIR| awk 'BEGIN {ORS=" "} {if(NR%100==0)print "."}'
