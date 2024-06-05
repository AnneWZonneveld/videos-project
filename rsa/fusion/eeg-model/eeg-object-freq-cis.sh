#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=fusion-cis                            
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=4
#SBATCH --mem=6000                             
#SBATCH --time=90:00:00                            
#SBATCH --qos=standard

# use array 1-3

# Set parameters
distance_type='euclidean-cv'
bin_width=0.1

# Extract the parameters
sub=$SLURM_ARRAY_TASK_ID

echo SLURM_ARRAY_JOB_ID: 
echo Subject: $sub
echo Distance type: $distance_type
echo Bin width: $bin_width

sleep 8

python eeg-object-freq-cis.py --sub $sub --distance_type $distance_type --jobarr_id $SLURM_ARRAY_TASK_ID --bin_width $bin_width