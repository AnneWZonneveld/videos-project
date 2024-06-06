#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=vp-plot                        
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=100
#SBATCH --time=00:30:00                   
#SBATCH --qos=standard

# nr of combinations = 0-2

# Set parameters
distance_type='pearson'
data_split='train'
model_metric='euclidean'

sleep 8

python vp_analysis_plot.py  --data_split $data_split --distance_type $distance_type  --model_metric $model_metric
