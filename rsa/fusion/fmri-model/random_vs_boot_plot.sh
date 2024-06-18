#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=fmri-plot                 
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=100                           
#SBATCH --time=00:30:00                            
#SBATCH --qos=standard

declare -a data_splits
declare -a distance_types
declare -a eval_methods

for s in 'test'; do
    for d in 'pearson'; do
            for e in 'spearman'; do
            data_splits[$index]=$s
            distance_types[$index]=$d
            features[$index]=$f
            eval_methods[$index]=$e
            ((index=index+1))
        done    
    done        
done        


# Extract the parameters
data_split=${data_splits[${SLURM_ARRAY_TASK_ID}]}
distance_type=${distance_types[${SLURM_ARRAY_TASK_ID}]}
eval_method=${eval_methods[${SLURM_ARRAY_TASK_ID}]}

echo Data split: $data_split
echo Distance type: $distance_type
echo Eval method:  $eval_method

python random_vs_boot_plot.py  --data_split $data_split  --distance_type $distance_type --eval_method $eval_method 