#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=avg-fmri               
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=7000                      
#SBATCH --time=8:00:00                         
#SBATCH --qos=standard


# Set parameters
rois='7AL,BA2,EBA,FFA,hV4,IPS0,IPS1-2-3,LOC,OFA,PFop,PFt,PPA,RSC,STS,TOS,V1d,V1v,V2d,V2v,V3ab,V3d,V3v'

# Create all relevant parameter combinations
declare -a data_splits 
declare -a distance_types

index=0

for s in 'test' 'train'; do
    for d in 'euclidean-cv'; do
        distance_types[$index]=$d
        data_splits[$index]=$s
        ((index=index+1))
    done
done

data_split=${data_splits[${SLURM_ARRAY_TASK_ID}]}
distance_type=${distance_types[${SLURM_ARRAY_TASK_ID}]}


# Extract the parameters
echo Distance type: $distance_type
echo data_split: $data_split

python GA-average-rdms.py --data_split $data_split --distance_type $distance_type --rois $rois
