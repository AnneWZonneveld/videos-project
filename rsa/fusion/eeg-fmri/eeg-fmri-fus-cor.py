"""
Performs EEG-fMRI fusion.
Calculating the similarity between specified ROI RDM and EEG RDMs for all time points.

Parameters
----------
zscore : int
	Whether to use z-scored EEG data [1] or not [0].
sfreq: int
    Sampling frequency of EEG data
data_split: str
    Train or test. 
eeg_distance_type: str
    Whether to use EEG RDMs based on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
fmri_distance_type: str
    Whether to use fMRI RDMs based on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
eval_method: str
    Method to compute similarity between RDMS; 'spearman' or 'pearson'
roi: str
    ROI.
jobarr_id: int
    Unique jobarray id.

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
from sklearn.cluster import KMeans, AgglomerativeClustering
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import argparse
from fusion_multiprocessing import * 

###################### Set parameters ###########################################
parser = argparse.ArgumentParser()
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--sfreq', default=500, type=int)
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--eeg_distance_type', default='pearson', type=str) 
parser.add_argument('--fmri_distance_type', default='pearson', type=str) 
parser.add_argument('--eval_method', default='spearman', type=str)
parser.add_argument('--roi', type=str) 
parser.add_argument('--jobarr_id', default=1, type=int) 
parser.add_argument('--n_cpus', default=1, type=int) 
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

########################### Loading in data #####################################
if args.data_split == 'train':
     n_conditions = 1000
elif args.data_split == 'test':
     n_conditions = 102

# EEG RDMs
print('Loading EEG rdms')
eeg_folder = f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_{args.zscore}/GA/'
eeg_file = eeg_folder + f'avg_{args.eeg_distance_type}_{format(args.sfreq, "04")}.pkl'
with open(eeg_file, 'rb') as f:
    eeg_data = pickle.load(f)
eeg_rdms = eeg_data['rdms_array']

# fmri RDMs
print('Loading fmri rdms')
fmri_folder = f'/scratch/azonneveld/rsa/fmri/rdms/GA/{args.fmri_distance_type}/'
fmri_file = fmri_folder + f'GA_{args.data_split}_rdm.pkl'
with open(fmri_file, 'rb') as f:
    fmri_data = pickle.load(f)
fmri_rdms = fmri_data['rdm_array']
roi_rdm = fmri_rdms[args.roi]
    
############################# Analysis ###########################################

# Calculate R^2
print('Starting multiprocessing R^2')
shm_name = f'GA_{args.roi}_{args.eeg_distance_type}_{args.fmri_distance_type}_{args.eval_method}_{args.data_split}_cor'
results = calc_rsquared_mp(eeg_rdms=eeg_rdms, roi_rdm=roi_rdm, jobarr_id=args.jobarr_id, its=10000, n_cpus=args.n_cpus, shm_name=shm_name, eval_method=args.eval_method)
print('Done multiprocessing')

############################ Save data #############################################

results_dict = {
    'data_split': args.data_split, 
    'eeg_distance_type': args.eeg_distance_type,
    'fmri_distance_type': args.fmri_distance_type,
    'roi': args.roi,
	'cor_values': results,
	'times': eeg_data['times'],
    'eval_method': args.eval_method
}

res_folder = f'/scratch/azonneveld/rsa/fusion/eeg-fmri/{args.data_split}/z_{args.zscore}/{args.roi}/fmri_{args.fmri_distance_type}/eeg_{args.eeg_distance_type}/{args.eval_method}/' 
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)

file_path = res_folder + f'cors.pkl'

# Save all model rdms
with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)


