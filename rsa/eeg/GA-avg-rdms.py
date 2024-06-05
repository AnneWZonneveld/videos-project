"""
Creating group level RDMs based on subject-level RDMs.

How to run example: 
    sbatch --array=1-2 create-rdms.sh 

Parameters
----------

zscore : int
	Whether to z-score [1] or not [0] the data.
data_split: str
    Train or test. 
distance_type: str
    Whether to base the RDMs on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
sfreq: int
    Sampling frequency. 
zoom: int
    For zoomed-in RDM y/n.

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
from scipy.stats import pearsonr, spearmanr
import argparse

###################### Set parameters ###########################################
subs = 6

parser = argparse.ArgumentParser()
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--distance_type', default='euclidean-cv', type=str)  
parser.add_argument('--sfreq', default=500, type=int)
parser.add_argument('--zoom', default=0, type=int)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

########################### Loading in data #####################################

# Model RDMs
print('Loading model rdms')
model_file = f'/scratch/azonneveld/rsa/model/rdms/t2/euclidean/rdm_t2_avg.pkl' 
with open(model_file, 'rb') as f: 
    model_rdms = pickle.load(f)
model_rdms = model_rdms[args.data_split]
feature_rdm = model_rdms['objects']

# Neural RDMs --> see if already available otherwise calculate
print('Loading neural rdms')
if args.zoom == 0:
    eeg_rdms_subs = np.zeros((subs, feature_rdm.shape[0], feature_rdm.shape[0], 185))
else:
    eeg_rdms_subs = np.zeros((subs, feature_rdm.shape[0], feature_rdm.shape[0], 301))

for i in range(subs):
    sub_format = format(i+1, '02')

    if args.zoom == True:
        eeg_file = f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_{args.zscore}/sub-{sub_format}/avg_{args.distance_type}_{format(args.sfreq, "04")}_zoom.pkl'
    else:
        eeg_file = f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_{args.zscore}/sub-{sub_format}/avg_{args.distance_type}_{format(args.sfreq, "04")}.pkl'

    with open(eeg_file, 'rb') as f: 
        eeg_data = pickle.load(f)
    eeg_rdms = eeg_data['rdms_array']
    eeg_rdms_subs[i, :, :, :] = eeg_rdms

GA_rdms = np.mean(eeg_rdms_subs, axis=0)

# Save GA rdms
GA_rdms_dict = {
    'data_split': args.data_split, 
    'distance_type': args.distance_type,
    'rdms_array': GA_rdms,
	'times': eeg_data['times']
}

if args.zoom == True:
    eeg_folder = f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_{args.zscore}/GA/zoom/'
else:
    eeg_folder = f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_{args.zscore}/GA/'
if not os.path.exists(eeg_folder) == True:
    os.makedirs(eeg_folder)

eeg_file = eeg_folder + f'avg_{args.distance_type}_{format(args.sfreq, "04")}.pkl'
with open(eeg_file, 'wb') as f:
    pickle.dump(GA_rdms_dict, f)

print('Done averaging RDMS')