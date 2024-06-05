"""
Creating zoomed in RDMs for EEG data based on data of specified subject --> 
this means with higher resulting sampling frequency and only until 1s.

How to run example: 
    sbatch --array=1-2 create-rdms.sh 

Parameters
----------
sub : int
	Used subject.
zscore : int
	Whether to z-score [1] or not [0] the data.
data_split: str
    Train or test. 
distance_type: str
    Whether to base the RDMs on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
batch: int
    Batch number (based on job array id) that determines which set of pseudotrials will be used (relevevant for cv-measurements)
sfreq: int
    Sampling frequency. 

"""


import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
from scipy.spatial.distance import squareform
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.svm import SVC
import argparse
import random
import concurrent.futures
from create_rdms_utils import compute_rdm
from functools import partial
from multiprocessing import shared_memory
from multiprocessing import current_process, cpu_count
from compute_rdms_mp import compute_rdms_multi


# Input argurments
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--data_split', default='train', type=str) 
parser.add_argument('--distance_type', default='euclidean', type=str) 
parser.add_argument('--batch', default=0, type=int) 
parser.add_argument('--sfreq', default=500, type=int)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results based on batch (a.k.a. permutation)
seed = args.batch
np.random.seed(seed)
n_cpus = 2

################################### Load EEG data ######################################
sub_format = format(args.sub, '02') 
sfreq_format = format(args.sfreq, '04') 

if args.sfreq == 50:
    data_dir = f'/scratch/giffordale95/projects/eeg_videos/dataset/preprocessed_data/dataset_02/eeg/sub-{sub_format}/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-{sfreq_format}/preprocessed_data.npy'
else:
    data_dir = f'/scratch/azonneveld/preprocessing/data/sub-{sub_format}/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-{sfreq_format}/preprocessed_data.npy'
    # data_dir = f'/scratch/azonneveld/preprocessing/sub-{sub_format}/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-0250/preprocessed_data.npy'

data_dict = np.load(data_dir, allow_pickle=True).item()
times = data_dict['times']
ch_names = data_dict['ch_names']
info = data_dict['info']
eeg_data_list = data_dict['eeg_data']
stimuli_presentation_order_list = data_dict['stimuli_presentation_order']
del data_dict

# Z score data per session
for session in range(len(eeg_data_list)):
    data_shape = eeg_data_list[session].shape
    data_provv = np.reshape(eeg_data_list[session], (len(eeg_data_list[session]), -1))
    if args.zscore == 1:
        scaler = StandardScaler()
        data_provv = scaler.fit_transform(data_provv)
    data_provv = np.reshape(data_provv, data_shape)
    if session == 0:
        eeg_data = data_provv
        stimuli_presentation_order = stimuli_presentation_order_list[session]
    else:   
        eeg_data = np.append(eeg_data, data_provv, 0)
        stimuli_presentation_order = np.append(stimuli_presentation_order, stimuli_presentation_order_list[session], 0)
    del data_provv
del eeg_data_list, stimuli_presentation_order_list

# Select the data split
if args.data_split == 'train':
    idx = np.where(stimuli_presentation_order <= 1000)[0]
elif args.data_split == 'test':
    idx = np.where(stimuli_presentation_order > 1000)[0]
eeg_data = eeg_data[idx]
stimuli_presentation_order = stimuli_presentation_order[idx]

# Temporally binning data (central aligned)
if args.sfreq != 50:

    if args.sfreq == 250:
        slide_size = 5
    elif args.sfreq == 500:
        slide_size = 2 #925 timepoints --> each timepoint is 4 ms --> 1200 ms, means 300 timepoints --> higher sampling freq than non-zoom
    
    # Ends at 1 s
    new_n_times = int(len(times)/slide_size)
    timepoint_dur = 3700/new_n_times
    time_end = int(1200/timepoint_dur)

    binned_eeg = np.zeros((eeg_data.shape[0], eeg_data.shape[1], time_end+1))
    for i in range(time_end + 1):
        if i == 0:
            binned_eeg[:, :, i] = np.mean(eeg_data[:, :, 0:int(slide_size/2)], axis=2)
        elif i == (len(times)-1):
            binned_eeg[:, :, i] = np.mean(eeg_data[:, :, (i-1)*slide_size:], axis=2)
        else:
            binned_eeg[:, :, i] = np.mean(eeg_data[:, :, (i-1)*slide_size : i*slide_size], axis=2)
    
    eeg_data = binned_eeg

    new_times = []
    for i in range(len(times)):
        if i % slide_size == 0:
            new_times.append(times[i])  
        if i == slide_size * time_end:
            break
    times = np.array(new_times)

    del binned_eeg

print('Loading EEG data done')

################################### Constructing RDMs #############################################
# Results array of shape:
# (Video conditions × Video conditions × EEG time points)
video_conditions = np.unique(stimuli_presentation_order)
n_conditions = video_conditions.shape[0]
n_channels = len(ch_names)
n_time = times.shape[0]
rdms_array = np.zeros((n_conditions, n_conditions, eeg_data.shape[2]), dtype='float32')

# Predetermine pseudotrials 
n_combinations = squareform(rdms_array[:, :, 0]).shape[0]
pseudo_order = np.zeros((n_combinations), dtype=object)
del rdms_array

random.seed(args.batch)
combination = 0
for v1 in range(n_conditions):
    for v2 in range(v1):
        idx_1 = resample(np.where(stimuli_presentation_order == video_conditions[v1])[0],replace=False)
        idx_2 = resample(np.where(stimuli_presentation_order == video_conditions[v2])[0],replace=False)
        ordering = (idx_1, idx_2)
        pseudo_order[combination] = ordering
        combination = combination + 1


# Parallel processing computing rdms
print('Starting multiprocessing')
results = compute_rdms_multi(eeg_data=eeg_data, pseudo_order=pseudo_order, ts=times, n_cpus=n_cpus, batch = args.batch, data_split=args.data_split, distance_type=args.distance_type, shm_name=f'{args.sub}_{args.sfreq}')
print('Done multiprocessing')

# Save 
results_dict = {
    'data_split': args.data_split, 
    'sub': args.sub, 
    'zscore': args.zscore,
    'distance_type': args.distance_type,
    'batch': args.batch,
	'rdms_array': results,
	'times': times,
	'ch_names': ch_names,
    'sfreq': args.sfreq,
	'info': info
}

res_folder = f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_{args.zscore}/sub-{sub_format}/{args.distance_type}/sfreq-{sfreq_format}/zoom/' 
if os.path.isdir(res_folder) == False:
	os.makedirs(res_folder)

file_path = res_folder + f'/{args.data_split}_{args.batch}_mp_zoom.pkl'

# Save all model rdms
with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)





