#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=eeg-fmri                    
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=1500            
#SBATCH --time=3-00:00:00                            
#SBATCH --qos=standard

declare -a data_splits
declare -a rois

eeg_distance_type='pearson'
fmri_distance_type='pearson'


for s in 'train'; do
    # for r in '7AL' 'BA2' 'EBA' 'FFA' 'hV4' 'IPS0' 'IPS1-2-3' 'LOC' 'OFA' 'PFop' 'PFt' 'PPA' 'RSC' 'STS' 'TOS' 'V1d' 'V1v' 'V2d' 'V2v' 'V3ab' 'V3d' 'V3v'; do
    for r in 'hV4' 'IPS0' 'V1v' 'V2d' 'V2v' 'V3ab' 'V3d' 'V3v'; do
        data_splits[$index]=$s
        rois[$index]=$r
        ((index=index+1))      
    done        
done

# Extract the parameters
data_split=${data_splits[${SLURM_ARRAY_TASK_ID}]}
roi=${rois[${SLURM_ARRAY_TASK_ID}]}

echo Data split: $data_split
echo Feature: $feature
echo Roi: $roi

python eeg-fmri-fus-cor.py  --data_split $data_split --roi $roi --eeg_distance_type $eeg_distance_type --fmri_distance_type $fmri_distance_type