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
# distance_type='euclidean-cv'
# sub=3
zscore=1
resampled=0
zoom=1
sfreq=500

# Create all relevant parameter combinations
declare -a sub_all
declare -a f_all
declare -a distance_type_all

index=0

for s in `seq 2 3`; do
    for f in 500; do
        for d in 'pearson'; do
            sub_all[$index]=$s
            f_all[$index]=$f
            distance_type_all[$index]=$d
            ((index=index+1))
        done
    done
done 

# # Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
sub=${sub_all[${SLURM_ARRAY_TASK_ID}]}
distance_type=${distance_type_all[${SLURM_ARRAY_TASK_ID}]}
sfreq=${f_all[${SLURM_ARRAY_TASK_ID}]}


echo Subject: $sub
echo Distance type: $distance_type
echo Z: $zscore
echo sfreq: $sfreq

sleep 8

python avg-rdms-batch.py --sub $sub --distance_type $distance_type  --zscore $zscore --sfreq $sfreq --data_split $data_split --resampled $resampled --zoom $zoom
