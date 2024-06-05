#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=fusion-cor                            
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000                             
#SBATCH --time=90:00:00                            
#SBATCH --qos=standard

# use array 0-6

# Set parameters
distance_type='euclidean-cv'
bin_width=0
zscore=1

# Create all relevant parameter combinations
declare -a feature_combs

feature_combs=('o-s-a' 'o-s' 'o-a' 's-a' 'o' 's' 'a')

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
feature=${feature_combs[${SLURM_ARRAY_TASK_ID}]}

echo Feature: $feature
echo Distance type: $distance_type

sleep 8

python eeg-model-fus-rw-ga.py --feature $feature --distance_type $distance_type --jobarr_id $SLURM_ARRAY_TASK_ID --bin_width $bin_width --zscore $zscore