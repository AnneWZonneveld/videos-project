#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=vp-plot                        
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=300
#SBATCH --time=00:30:00                        
#SBATCH --qos=standard

# nr of combinations = 0-2

# Set parameters
distance_type='pearson'
zscore=1
sfreq=500

# Create all relevant parameter combinations
declare -a sub_all

index=0

for s in `seq 1 3` ; do
    sub_all[$index]=$s
    ((index=index+1))
done 

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
sub=${sub_all[${SLURM_ARRAY_TASK_ID}]}

echo Subject: $sub
echo Distance type: $distance_type

sleep 8

python vp-analysis-plot.py --sub $sub --distance_type $distance_type  --zscore $zscore --sfreq $sfreq