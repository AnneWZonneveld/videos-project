#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=descriptives                      
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000                     
#SBATCH --time=8:00:00                            
#SBATCH --qos=prio

zscore=1

# Create all relevant parameter combinations
declare -a data_splits
declare -a sub_all
declare -a sfreq_all
declare -a distance_type_all
declare -a eval_method_all

index=0

for ds in 'train'; do
    for s in `seq 1 3`; do
        for sf in 500 ; do
            for d in 'pearson'; do
                for e in 'spearman'; do
                    data_splits[$index]=$ds
                    sub_all[$index]=$s
                    feature_all[$index]=$f
                    sfreq_all[$index]=$sf
                    distance_type_all[$index]=$d
                    eval_method_all[$index]=$e
                    ((index=index+1))
                done
            done    
        done        
    done        
done


# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
data_split=${data_splits[${SLURM_ARRAY_TASK_ID}]}
sub=${sub_all[${SLURM_ARRAY_TASK_ID}]}
sfreq=${sfreq_all[${SLURM_ARRAY_TASK_ID}]}
distance_type=${distance_type_all[${SLURM_ARRAY_TASK_ID}]}
eval_method=${eval_method_all[${SLURM_ARRAY_TASK_ID}]}


echo Subject: $sub
echo Distance type: $distance_type

python descriptives.py --sub $sub --zscore $zscore --data_split $data_split --distance_type $distance_type --sfreq $sfreq