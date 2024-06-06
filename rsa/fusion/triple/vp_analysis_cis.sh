#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=triple-cis                  
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=16
#SBATCH --mem=3500        
#SBATCH --time=6-00:00:00                            
#SBATCH --qos=standard


declare -a data_splits
declare -a rois
declare -a features_oi

eeg_distance_type='pearson'
fmri_distance_type='pearson'
model_metric='euclidean'

for s in 'train'; do
    for r in 'EBA' 'PPA' 'LOC'; do
        for f in 'objects' 'scenes' 'actions'; do 
            data_splits[$index]=$s
            features_oi[$index]=$f
            rois[$index]=$r
            ((index=index+1))
        done        
    done        
done

# Extract the parameters
data_split=${data_splits[${SLURM_ARRAY_TASK_ID}]}
feature_oi=${features_oi[${SLURM_ARRAY_TASK_ID}]}
roi=${rois[${SLURM_ARRAY_TASK_ID}]}

echo vp-analysis-cis
echo Data split: $data_split
echo Feature: $feature
echo Roi: $roi

python vp_analysis_cis.py  --data_split $data_split --feature_oi $feature_oi --roi $roi --eeg_distance_type $eeg_distance_type --fmri_distance_type $fmri_distance_type  --n_cpus ${SLURM_CPUS_PER_TASK}