#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=fmri-rdm                    
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=4
#SBATCH --mem=6000                 
#SBATCH --time=10:00:00                         
#SBATCH --qos=standard


# Set parameters
data_split='test'
# distance_type='pearson'
rois='7AL,BA2,EBA,FFA,hV4,IPS0,IPS1-2-3,LOC,OFA,PFop,PFt,PPA,RSC,STS,TOS,V1d,V1v,V2d,V2v,V3ab,V3d,V3v'
batch=0

# Create all relevant parameter combinations
declare -a roi_all
declare -a distance_types
declare -a sub_all

index=0

for s in `seq 1 10`; do
    for d in 'euclidean-cv'; do
        sub_all[$index]=$s
        distance_types[$index]=$d
        ((index=index+1))
    done
done

# Extract the parameters
# echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
sub=${sub_all[${SLURM_ARRAY_TASK_ID}]}
distance_type=${distance_types[${SLURM_ARRAY_TASK_ID}]}
batch=$batch

echo Distance type: $distance_type
echo sub: $sub
echo rois: $rois
echo split: $data_split

python create-rdms.py --sub $sub --rois $rois --distance_type $distance_type --data_split $data_split --batch $batch
