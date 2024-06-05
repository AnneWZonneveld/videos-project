#!/bin/bash
#SBATCH --job-name=preprocessing_eeg_videos
#SBATCH --mem=9000
#SBATCH --time=10:00:00
#SBATCH --qos=prio

## array of 0-2

# Creating the parameters combinations
declare -a sub_all
declare -a baseline_correction_all
declare -a highpass_all
declare -a mvnn_all
index=0

for s in `seq 5 6` ; do
	for b in '1' ; do
		for h in '0.01' ; do
			for m in 'time' ; do
				sub_all[$index]=$s
				baseline_correction_all[$index]=$b
				highpass_all[$index]=$h
				mvnn_all[$index]=$m
				((index=index+1))
			done
		done
	done
done

# Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
sub=${sub_all[$SLURM_ARRAY_TASK_ID]}
baseline_correction=${baseline_correction_all[$SLURM_ARRAY_TASK_ID]}
highpass=${highpass_all[$SLURM_ARRAY_TASK_ID]}
mvnn=${mvnn_all[$SLURM_ARRAY_TASK_ID]}
sfreq=500
echo Subject: $sub
echo Baseline correction: $baseline_correction
echo Highpass: $highpass
echo MVNN: $mvnn

sleep 8

# Running the job
python preprocessing.py --sub $sub --baseline_correction $baseline_correction --mvnn $mvnn --highpass $highpass --sfreq $sfreq 