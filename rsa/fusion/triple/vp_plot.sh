#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=triple-plot                   
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=100                  
#SBATCH --time=00:30:00                            
#SBATCH --qos=standard


declare -a data_splits
declare -a rois
declare -a features_oi

eeg_distance_type='pearson'
fmri_distance_type='pearson'
ceiling=0

index = 0
for s in 'train'; do
    for r in 'EBA' 'LOC' 'PPA' '7AL' 'BA2' 'FFA' 'hV4' 'IPS0' 'IPS1-2-3' 'OFA' 'PFop' 'PFt' 'RSC' 'STS' 'TOS' 'V1d' 'V1v' 'V2d' 'V2v' 'V3ab' 'V3d' 'V3v'; do
    # for r in '7AL' 'BA2' 'IPS0' 'IPS1-2-3' 'PFop' 'PFt' 'V3ab'; do
        data_splits[$index]=$s
        rois[$index]=$r
        ((index=index+1))    
    done        
done

# Extract the parameters
data_split=${data_splits[${SLURM_ARRAY_TASK_ID}]}
roi=${rois[${SLURM_ARRAY_TASK_ID}]}

echo Data split: $data_split
echo Roi: $roi

python vp_plot.py  --data_split $data_split --roi $roi --eeg_distance_type $eeg_distance_type --fmri_distance_type $fmri_distance_type --ceiling $ceiling