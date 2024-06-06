#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=vp-cis                         
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=32
#SBATCH --mem=6000
#SBATCH --time=5-00:00:00                 
#SBATCH --qos=standard

# nr of combinations = length job array  = 3 * 3 = 9 --> use array 0-8

# Set parameters
distance_type='pearson'
zscore=1
sfreq=500
data_split='train'


echo Distance type: $distance_type
echo Data split: $data_split

sleep 8

python vp_analysis_ga_cis.py --data_split $data_split --distance_type $distance_type --jobarr_id $SLURM_ARRAY_TASK_ID --zscore $zscore --sfreq $sfreq --n_cpus  ${SLURM_CPUS_PER_TASK}