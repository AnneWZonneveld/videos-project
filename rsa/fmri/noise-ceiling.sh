#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=noise-ceiling             
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=500                    
#SBATCH --time=03:00:00                            
#SBATCH --qos=standard

declare -a data_splits
declare -a distance_types
declare -a rois

# n batches  = 88

for s in 'train' 'test'; do
    for d in 'euclidean' 'pearson'; do
        # for r in '7AL' 'BA2' 'EBA' 'FFA' 'hV4' 'IPS0' 'IPS1-2-3' 'LOC' 'OFA' 'PFop' 'PFt' 'PPA' 'RSC' 'STS' 'TOS' 'V1d' 'V1v' 'V2d' 'V2v' 'V3ab' 'V3d' 'V3v'; do
        for r in 'RSC' 'TOS'; do
            data_splits[$index]=$s
            distance_types[$index]=$d
            rois[$index]=$r
            ((index=index+1))
        done    
    done        
done        

# Extract the parameters
data_split=${data_splits[${SLURM_ARRAY_TASK_ID}]}
distance_type=${distance_types[${SLURM_ARRAY_TASK_ID}]}
roi=${rois[${SLURM_ARRAY_TASK_ID}]}

echo Data split: $data_split
echo Distance type: $distance_type
echo Roi: $roi

python noise-ceiling.py  --data_split $data_split  --distance_type $distance_type --roi $roi