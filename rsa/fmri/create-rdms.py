"""
Creating RDMs for fmri data based on different distance measurements.

How to run example: 
    sbatch --array=1-2 create-rdms.sh 

Parameters
----------
sub : int
	Used subject.
zscore : int
	Whether to z-score [1] or not [0] the data.
data_split : str
	Whether to decode the 'testing' or 'training' split.
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
from scipy.spatial.distance import squareform
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.svm import SVC
import argparse
from compute_rdms_mp import compute_rdms_multi

n_cpus = 2

# Input argurments
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--data_split', default='train', type=str) 
parser.add_argument('--distance_type', default='euclidean', type=str) 
parser.add_argument('--rois', type=str)
parser.add_argument('--batch', type=int, default=0)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


################################### Load fMRI data ######################################
sub_format = format(args.sub, '02') 
rois = args.rois.split(',')
if args.data_split == 'train':
    data_split_format = 'training'
elif args.data_split == 'test':
    data_split_format = 'testing'

data_dir = f'/scratch/giffordale95/projects/bold_moments/fmri_time_2/fmri_dataset/reliable_preprocessed_data/sub{sub_format}/{data_split_format}'

rois_data = {}
for roi in rois:

    try:
        fmri_file = data_dir + f'/{roi}_TRavg-56789_{data_split_format}.pkl'
        with open(fmri_file, 'rb') as f: 
            data = pickle.load(f)
        if args.data_split == 'train':
            data = data['train']
        elif args.data_split == 'test':
            data = data['test_data']

        # zscore across conditions
        sc = StandardScaler()
        data = sc.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

        rois_data[roi] = data
    except:
         print(f"No data for {roi}")
         print(f"{data.shape}")

print('Loading fMRI data done')

################################### Constructing RDMs #############################################


# Results array of shape:
# (Video conditions × Video conditions)
n_conditions = data.shape[0]
n_reps = data.shape[1]
rdm_array = np.zeros((n_conditions, n_conditions), dtype='float32')

# Determine pseudotrials
seed = args.batch
np.random.seed(seed)

n_combinations = squareform(rdm_array).shape[0]
pseudo_order = np.zeros((n_combinations), dtype=object)
del rdm_array

combination = 0
for v1 in range(n_conditions):
    for v2 in range(v1):
        idx_1 = resample(range(n_reps),replace=False)
        idx_2 = resample(range(n_reps),replace=False)
        ordering = (idx_1, idx_2)
        pseudo_order[combination] = ordering
        combination = combination + 1


# Parallel processing computing rdms
print('Starting multiprocessing')
shm_name = f'{args.sub}_{args.data_split}_{args.batch}'
results = compute_rdms_multi(fmri_data=rois_data, pseudo_order=pseudo_order, batch=args.batch, data_split=args.data_split, n_cpus=n_cpus, distance_type=args.distance_type, shm_name=shm_name)
print('Done multiprocessing')

# Restructure results
roi_res = {}
for i in range(len(results)):
    key = results[i][0]
    rdm = results[i][1]
    roi_res[key] = rdm

# Save 
results_dict = {
    'data_split': args.data_split, 
    'results': roi_res,
    'sub': args.sub,
    'distance_type': args.distance_type
}

res_folder = f'/scratch/azonneveld/rsa/fmri/rdms/sub-{sub_format}/{args.distance_type}/'                                                                                                                            
if os.path.isdir(res_folder) == False:
	os.makedirs(res_folder)

file_path = res_folder + f'/{args.data_split}_rdms.pkl'

# Save all model rdms
with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)





