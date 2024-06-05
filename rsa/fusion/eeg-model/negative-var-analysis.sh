#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=negative-var                       
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000                            
#SBATCH --time=00:01:00                      
#SBATCH --qos=standard

# Set parameters
sub=3
slide=0
ridge=0
cv=0
zscore=1

sleep 8

python negative-var-analysis.py --sub $sub --slide $slide --ridge $ridge --cv_r2 $cv --zscore $zscore