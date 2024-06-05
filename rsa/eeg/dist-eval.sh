#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=dist-eval                            
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000                           
#SBATCH --time=00:30:00                         
#SBATCH --qos=standard


# Set parameters
distance_type='euclidean-cv'

# Create all relevant parameter combinations
declare -a f_all
declare -a z_all

index=0

for f in 50 500; do
    for z in 0 1; do
        sub_all[$index]=$s
        f_all[$index]=$f
        z_all[$index]=$z
        ((index=index+1))
    done
done


# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
sub=${sub_all[${SLURM_ARRAY_TASK_ID}]}
zscore=${z_all[${SLURM_ARRAY_TASK_ID}]}
sfreq=${f_all[${SLURM_ARRAY_TASK_ID}]}


echo Distance type: $distance_type
echo Z: $zscore
echo sfreq: $sfreq

python dist-eval.py --distance_type $distance_type --zscore $zscore --sfreq $sfreq

