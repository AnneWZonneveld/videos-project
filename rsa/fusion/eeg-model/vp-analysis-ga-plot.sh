#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=vp-plot                        
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=300
#SBATCH --time=00:30:000                 
#SBATCH --qos=standard

# nr of combinations = 0-2

# Set parameters
distance_type='pearson'
zscore=1
sfreq=500
data_split='train'


echo Subject: $sub
echo Distance type: $distance_type

sleep 8

python vp-analysis-ga-plot.py  --data_split $data_split --distance_type $distance_type  --zscore $zscore --sfreq $sfreq 