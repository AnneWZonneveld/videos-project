"""
Creating RDMs for EEG data based on different distance measurements.

How to run example: 
    sbatch --array=1-2 create-rdms.sh 

Parameters
----------
sub : int
	Used subject.
zscore : int
	Whether to z-score [1] or not [0] the data.
data_split : str
	Whether to decode the 'test' or 'train' split.
distance_type: str
    Whether to base the RDMs on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
batch: int
    Batch number (based on job array id) that determines which set of permutations will be executed. 
    We will run 10 permutations --> 1 batch for every permutation
"""


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
from sklearn.svm import SVC
import argparse
import concurrent.futures
from create_rdms_utils import compute_rdm
from functools import partial
from multiprocessing import shared_memory
from multiprocessing import current_process, cpu_count


# Input argurments
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--data_split', default='train', type=str) 
parser.add_argument('--distance_type', default='euclidean', type=str) 
parser.add_argument('--batch', default=0, type=int) 
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# Set random seed for reproducible results based on batch (a.k.a. permutation)
seed = args.batch
np.random.seed(seed)


################################### Load EEG data ######################################
sub_format = format(args.sub, '02') 
data_dir = f'/scratch/giffordale95/projects/eeg_videos/dataset/preprocessed_data/dataset_02/eeg/sub-{sub_format}/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-0050/preprocessed_data.npy'
# data_dir = f'/scratch/giffordale95/projects/eeg_videos/dataset/preprocessed_data/dataset_02/eeg/sub-{sub}/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-0050/preprocessed_data.npy'
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
    if args.zscore == 1:
    # if zscore == 1:
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
# if data_split == 'train':
    idx = np.where(stimuli_presentation_order <= 1000)[0]
elif args.data_split == 'test':
    idx = np.where(stimuli_presentation_order > 1000)[0]
eeg_data = eeg_data[idx]
stimuli_presentation_order = stimuli_presentation_order[idx]


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

combination = 0
for v1 in range(n_conditions):
    for v2 in range(v1):
        idx_1 = resample(np.where(stimuli_presentation_order == video_conditions[v1])[0],replace=False)
        idx_2 = resample(np.where(stimuli_presentation_order == video_conditions[v2])[0],replace=False)
        ordering = (idx_1, idx_2)
        pseudo_order[combination] = ordering
        combination = combination + 1


# Creating shared memory
shm_name = 'shared_data_permute'

try:
    shm = shared_memory.SharedMemory(create=True, size=rdms_array.nbytes, name=shm_name)
except FileExistsError:
    shm_old = shared_memory.SharedMemory(shm_name, create=False)
    shm_old.close()
    shm_old.unlink()
    shm = shared_memory.SharedMemory(create=True, size=rdms_array.nbytes, name=shm_name)

# Create a np.recarray using the buffer of shm
shm_rdms_array = np.recarray(shape=rdms_array.shape, dtype=rdms_array.dtype, buf=shm.buf)

# Copy the data into the shared memory
np.copyto(shm_rdms_array, rdms_array)


# Parrallel calculating of RDMs for timepoints
partial_compute_rdm = partial(compute_rdm,
                               eeg_data=eeg_data,
                               pseudo_order=pseudo_order,
                               data_split=args.data_split,
                               distance_type= args.distance_type)

with concurrent.futures.ProcessPoolExecutor(cpu_count) as executor:
    ts = range(eeg_data.shape[2])
    results = executor.map(partial_compute_rdm, ts)

# Format data into array
for t in range(eeg_data.shape[2]):
    rdm = results[t]
    rdms_array[:, :, t] = rdm


# Save results
results_dict = {
    'data_split': args.data_split, 
    'sub': args.sub, 
    'zscore': args.zscore,
    'distance_type': args.distance_type,
    'batch': args.batch,
	'rdms_array': rdms_array,
	'times': times,
	'ch_names': ch_names,
	'info': info
}

res_folder = f'/scratch/azonneveld/rsa/eeg/rdms/sub-{sub_format}/{args.distance_type}' 
if not os.path.exists(res_folder) == True:
    os.mkdir(res_folder)

file_path = res_folder + f'/{args.data_split}_{args.batch}.pkl'

# Save all model rdms
with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)





