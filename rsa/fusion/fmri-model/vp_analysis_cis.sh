#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=vp-cis                    
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=8
#SBATCH --mem=500 
#SBATCH --time=3-00:00:00                            
#SBATCH --qos=standard

# nr of combinations = length job array  = 3 * 3 = 9 --> use array 0-8

# Set parameters
distance_type='pearson'
zscore=1
sfreq=500
data_split='test'
rois='7AL,BA2,EBA,FFA,hV4,IPS0,IPS1-2-3,LOC,OFA,PFop,PFt,PPA,RSC,STS,TOS,V1d,V1v,V2d,V2v,V3ab,V3d,V3v'
model_metric='euclidean'


echo Distance type: $distance_type
echo Data split: $data_split

sleep 8

python vp_analysis_cis.py --rois $rois --data_split $data_split --distance_type $distance_type --jobarr_id $SLURM_ARRAY_TASK_ID --zscore $zscore --n_cpus  ${SLURM_CPUS_PER_TASK} --model_metric $model_metric