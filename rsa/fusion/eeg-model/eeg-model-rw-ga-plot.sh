#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=vp-plot                          
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000                            
#SBATCH --time=00:10:00                            
#SBATCH --qos=standard

zscore=1

python eeg-model-rw-ga-plot.py --zscore $zscore