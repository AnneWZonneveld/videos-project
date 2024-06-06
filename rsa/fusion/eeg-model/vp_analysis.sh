#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=vp                         
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=16
#SBATCH --mem=2000                             
#SBATCH --time=3-00:00:00                            
#SBATCH --qos=standard

# nr of combinations = length job array  = 3 * 3 = 9 --> use array 0-8

# Set parameters
distance_type='euclidean-cv'
zscore=0
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
echo Feature: $feature
echo Distance type: $distance_type

sleep 8

python vp_analysis.py --sub $sub --distance_type $distance_type --jobarr_id $SLURM_ARRAY_TASK_ID --zscore $zscore --sfreq $sfreq