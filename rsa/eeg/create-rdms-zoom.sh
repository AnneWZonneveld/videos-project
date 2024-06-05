#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=create-rdms-z                             
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=2
#SBATCH --mem=10000                        
#SBATCH --time=90:00:00                            
#SBATCH --qos=standard

sub=2
zscore=1
data_split='test'
distance_type='pearson'
sfreq=500

echo Subject: $sub
echo Distance type: $distance_type
echo Permutation: $SLURM_ARRAY_TASK_ID

python create-rdms-zoom.py --sub $sub --zscore $zscore --data_split $data_split --distance_type $distance_type --batch $SLURM_ARRAY_TASK_ID --sfreq $sfreq