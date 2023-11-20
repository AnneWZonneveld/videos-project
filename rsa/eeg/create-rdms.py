import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.utils import resample
import tensorflow as tf
import tensorflow_hub as hub
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Parameters
sub = '01'
zscore = True
data_split = 'train'
distance_type = 'euclidean'

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)

# Helper functions
def euclidean_distance(a, b):
    dist = np.linalg.norm(a-b)
    return dist

################################### Load EEG data ######################################
data_dir = f'/scratch/giffordale95/projects/eeg_videos/dataset/preprocessed_data/dataset_02/eeg/sub-{sub}/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-0050/preprocessed_data.npy'
data_dict = np.load(data_dir, allow_pickle=True).item()
times = data_dict['times']
ch_names = data_dict['ch_names']
info = data_dict['info']
eeg_data_list = data_dict['eeg_data']
stimuli_presentation_order_list = data_dict['stimuli_presentation_order']

# Z score data per session
for session in range(len(eeg_data_list)):
    data_shape = eeg_data_list[session].shape
    data_provv = np.reshape(eeg_data_list[session], (len(eeg_data_list[session]), -1))
    if zscore == True:
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
if data_split == 'train':
    idx = np.where(stimuli_presentation_order <= 1000)[0]
elif data_split == 'test':
    idx = np.where(stimuli_presentation_order > 1000)[0]
eeg_data = eeg_data[idx]
stimuli_presentation_order = stimuli_presentation_order[idx]


################################### Constructing RDMs #############################################
# Results array of shape:
# (Video conditions × Video conditions × EEG time points)

n_perm = 20
video_conditions = np.unique(stimuli_presentation_order)
n_conditions = video_conditions.shape[0]
n_channels = len(ch_names)
n_time = times.shape[0]
rdms_array = np.zeros((n_conditions, n_conditions, eeg_data.shape[2]))

if distance_type == 'euclidean':
    distance_measure = euclidean_distance
# to do: other distance measurements


# Loop over EEG time points, image-conditions and EEG repetitions
for t in tqdm(range(eeg_data.shape[2])):

    # TO DO: implement for loop for n of permutations? 

    for v1 in range(n_conditions):
        for v2 in range(v1):
            idx_1 = resample(np.where(stimuli_presentation_order == video_conditions[v1])[0],replace=False)
            idx_2 = resample(np.where(stimuli_presentation_order == video_conditions[v2])[0],replace=False)
            eeg_cond_1 = eeg_data[idx_1,:,t]
            eeg_cond_2 = eeg_data[idx_2,:,t]

            # Select a minimum amount of trials in case repeats are not the 
            # same between two test image conditions
            if len(eeg_cond_1) < len(eeg_cond_2):
                eeg_cond_2 = eeg_cond_2[:len(eeg_cond_1)]
            elif len(eeg_cond_2) < len(eeg_cond_1):
                eeg_cond_1 = eeg_cond_1[:len(eeg_cond_2)]
            
            # Create pseudo-trials
            if data_split == 'test':
                n_ptrials_repeats = 4
            elif data_split == 'train':
                n_ptrials_repeats = 2
            n_pseudo_trials = int(np.ceil(len(eeg_cond_1) / n_ptrials_repeats))
            pseudo_data_1 = np.zeros((n_pseudo_trials, eeg_cond_1.shape[1]))
            pseudo_data_2 = np.zeros((n_pseudo_trials, eeg_cond_2.shape[1]))

            for r in range(n_pseudo_trials):
                idx_start = r * n_ptrials_repeats
                idx_end = idx_start + n_ptrials_repeats
                pseudo_data_1[r] = np.mean(eeg_cond_1[idx_start:idx_end],0)
                pseudo_data_2[r] = np.mean(eeg_cond_2[idx_start:idx_end],0)
            
            eeg_cond_1 = pseudo_data_1
            eeg_cond_2 = pseudo_data_2

            if distance_type == 'euclidean':
                eeg_cond_1 = np.mean(eeg_cond_1,0)
                eeg_cond_2 = np.mean(eeg_cond_2,0)
                distance = np.linalg.norm(eeg_cond_1-eeg_cond_2)

                rdms_array[v1, v2, t] = distance
                rdms_array[v2, v1, t] = distance


# Save results
results_dict = {
    'n_perm': n_perm,
    'data_split': data_split, 
    'sub': sub, 
    'zscore': zscore,
    'distance_type': distance_type,
	'rdms_array': rdms_array,
	'times': times,
	'ch_names': ch_names,
	'info': info
}

res_folder = f'/scratch/azonneveld/rsa/eeg/rdms/{sub}/' 
file_path = res_folder + f'{distance_type}.pkl'

# Save all model rdms
with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)





