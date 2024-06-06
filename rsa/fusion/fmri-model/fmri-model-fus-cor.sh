#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=fmri-cor                      
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=4
#SBATCH --mem=1000     
#SBATCH --time=90:00:00                            
#SBATCH --qos=standard

# 0-59 batch

# data_split='test'
# distance_type='pearson'
# eval_method='spearman'
# feature='objects'
rois='7AL,BA2,EBA,FFA,hV4,IPS0,IPS1-2-3,LOC,OFA,PFop,PFt,PPA,RSC,STS,TOS,V1d,V1v,V2d,V2v,V3ab,V3d,V3v'

declare -a subjects
declare -a data_splits
declare -a distance_types
declare -a features
declare -a eval_methods

for s in `seq 1 10`; do
    for ds in 'train'; do
        for d in 'pearson'; do
            for f in 'objects' 'scenes' 'actions' ; do 
                for e in 'spearman'; do
                    subjects[$index]=$s
                    data_splits[$index]=$ds
                    distance_types[$index]=$d
                    features[$index]=$f
                    eval_methods[$index]=$e
                    ((index=index+1))
                done
            done    
        done        
    done        
done

# Extract the parameters
sub=${subjects[${SLURM_ARRAY_TASK_ID}]}
data_split=${data_splits[${SLURM_ARRAY_TASK_ID}]}
distance_type=${distance_types[${SLURM_ARRAY_TASK_ID}]}
feature=${features[${SLURM_ARRAY_TASK_ID}]}
eval_method=${eval_methods[${SLURM_ARRAY_TASK_ID}]}

echo Data split: $data_split
echo Distance type: $distance_type
echo Feature: $feature
echo Eval method:  $eval_method

python fmri-model-fus-cor.py --sub $sub --data_split $data_split --feature $feature --distance_type $distance_type --eval_method $eval_method --rois $rois --n_cpus ${SLURM_CPUS_PER_TASK}