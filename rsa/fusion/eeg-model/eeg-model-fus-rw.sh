#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=fusion-cor                            
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=4
#SBATCH --mem=4000                             
#SBATCH --time=100:00:00                            
#SBATCH --qos=standard

# use array 0-83
# use array 0-20

# Set parameters
distance_type='euclidean-cv'
bin_width=0
zscore=1
slide=1

# Create all relevant parameter combinations
declare -a features_all
declare -a sub_all
declare -a ridge_all
declare -a cv_all
index=0

# for s in `seq 1 3`; do
# 	for f in 'o-s-a' 'o-s' 'o-a' 's-a' 'o' 's' 'a' ; do
# 		for r in 0 1 ; do
# 			for cv in 0 1 ; do
# 				sub_all[$index]=$s
# 				features_all[$index]=$f
# 				ridge_all[$index]=$r
# 				cv_all[$index]=$cv
# 				((index=index+1))
# 			done
# 		done
# 	done
# done


# # Extract the parameters
# echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
# feature=${features_all[${SLURM_ARRAY_TASK_ID}]}
# sub=${sub_all[${SLURM_ARRAY_TASK_ID}]}
# ridge=${ridge_all[${SLURM_ARRAY_TASK_ID}]}
# cv=${cv_all[${SLURM_ARRAY_TASK_ID}]}

for s in `seq 1 3`; do
	for f in 'o-s-a' 'o-s' 'o-a' 's-a' 'o' 's' 'a' ; do
		sub_all[$index]=$s
		features_all[$index]=$f
		((index=index+1))
	done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
feature=${features_all[${SLURM_ARRAY_TASK_ID}]}
sub=${sub_all[${SLURM_ARRAY_TASK_ID}]}
ridge=0
cv=0

echo Subject: $sub
echo Feature: $feature
echo Distance type: $distance_type

sleep 8

python eeg-model-fus-rw.py --sub $sub --feature $feature --distance_type $distance_type --jobarr_id $SLURM_ARRAY_TASK_ID --bin_width $bin_width --slide $slide --zscore $zscore --ridge $ridge --cv_r2 $cv