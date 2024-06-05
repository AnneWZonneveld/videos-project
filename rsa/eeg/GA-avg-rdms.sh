#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=avg-rdms                               
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000                       
#SBATCH --time=24:00:00                      
#SBATCH --qos=standard


# Set parameters
data_split='train'
zscore=1
resampled=0
zoom=0
sfreq=500
distance_type='pearson'


echo Split: $data_split
echo Distance type: $distance_type
echo Z: $zscore
echo sfreq: $sfreq

sleep 8

python GA-avg-rdms.py --distance_type $distance_type  --zscore $zscore --sfreq $sfreq --data_split $data_split --zoom $zoom
