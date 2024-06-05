#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=vp-plot                          
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000                            
#SBATCH --time=00:10:00                            
#SBATCH --qos=standard

zscore=1


# Create all relevant parameter combinations --> array 0-11
declare -a sub_all
declare -a ridge_all
declare -a cv_all

index=0

# for s in `seq 1 3` ; do
#     for r in 0 1 ; do 
#         for c in 0 1; do
#             sub_all[$index]=$s
#             ridge_all[$index]=$r
#             cv_all[$index]=$c
#             ((index=index+1))
#         done
#     done
# done 

# for s in `seq 1 3` ; do
#     for c in 0 1; do
#         sub_all[$index]=$s
#         cv_all[$index]=$c
#         ((index=index+1))
#     done
# done

for s in `seq 1 3` ; do
    sub_all[$index]=$s
    ((index=index+1))
done

sub=${sub_all[${SLURM_ARRAY_TASK_ID}]}
# cv=${cv_all[${SLURM_ARRAY_TASK_ID}]}
# ridge=${ridge_all[${SLURM_ARRAY_TASK_ID}]}
cv=0
ridge=0
slide=1

python eeg-model-rw-plot.py --sub $sub --zscore $zscore --cv_r2 $cv --ridge $ridge --slide $slide