#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=perm-inspect               
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --time=10:00:00                     
#SBATCH --qos=standard


# Set parameters
distance_type='pearson'
zscore=1
sfreq=500
data_split='train'
rois='LOC'
model_metric='euclidean'

echo Distance type: $distance_type
echo Data split: $data_split

sleep 8

python permutation_inspection.py --rois $rois --data_split $data_split --distance_type $distance_type  --n_cpus  ${SLURM_CPUS_PER_TASK} --model_metric $model_metric