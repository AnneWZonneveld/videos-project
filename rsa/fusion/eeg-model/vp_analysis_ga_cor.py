"""
Performs main variance partitioning on group level.

Parameters
----------
data_split: str
    Train or test. 
zscore : int
	Whether to use z-scored EEG data [1] or not [0].
sfreq: int
    Sampling frequency of EEG data
distance_type: str
    Whether to use EEG RDMs based on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
jobarr_id: int
    Unique jobarray id.
model_metric: str
    Metric used in the model RDM; 'pearson'/'euclidean'
n_cpus: int
    Number of cpus. 

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
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import fdrcorrection
from functools import partial
import multiprocessing as mp
from vp_utils import *
from vp_multiprocessing import *

###################### Set parameters ###########################################
parser = argparse.ArgumentParser()
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--distance_type', default='pearson', type=str) 
parser.add_argument('--jobarr_id', default=1, type=int) 
parser.add_argument('--sfreq', default=500, type=int)
parser.add_argument('--model_metric', default='euclidean', type=str)
parser.add_argument('--n_cpus', default=1, type=int)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


########################### Loading in data #####################################

# Model RDMs
print('Loading model rdms')
model_file = f'/scratch/azonneveld/rsa/model/rdms/t2/{args.model_metric}/rdm_t2_avg.pkl' #change this to permuted freq matrix? / avg
with open(model_file, 'rb') as f: 
    model_rdms = pickle.load(f)
model_rdms = model_rdms[args.data_split]

# Neural RDMs
print('Loading neural rdms')
eeg_file = f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_{args.zscore}/GA/avg_{args.distance_type}_{format(args.sfreq, "04")}.pkl'
with open(eeg_file, 'rb') as f: 
    eeg_data = pickle.load(f)
eeg_rdms = eeg_data['rdms_array']

######################## Reweighting of RDMs #######################################
models = ['o', 's', 'a', 'o-a', 'o-s', 'o-a', 's-a', 'o-s-a']
sq_len = squareform(model_rdms['objects'], checks=False).shape[0]

design_matrices = {}
for m in models:
    vars = m.split('-')
    design_matrix = np.zeros((sq_len, len(vars)))
    
    for i in range(len(vars)):
        v = vars[i]

        if v == 'o':
            var = 'objects'
        elif v == 's':
            var = 'scenes'
        elif v == 'a':
            var = 'actions'
    
        sq_rdm = squareform(model_rdms[var], checks=False).squeeze()
        design_matrix[:, i] = sq_rdm 

    design_matrices[m] = design_matrix

    
# Calculate R^2
print('Starting multiprocessing reweighted R^2')
results = calc_rsquared_rw2_mp(eeg_rdms=eeg_rdms, design_matrices=design_matrices, jobarr_id=args.jobarr_id, n_cpus=args.n_cpus, shm_name=f'vp_eeg_GA_{args.data_split}', its=10000)
print('Done multiprocessing')

# Restructure
values_rs = {
        'u_a': [],
        'u_s': [],
        'u_o': [],
        'os_shared' : [],
        'sa_shared' : [],
        'oa_shared' : [],
        'osa_shared' : [],
        'o_total': [],
        'a_total' : [],
        's_total' : [],
        'all_parts': []
    }

ps_rs = {
        'u_a': [],
        'u_s': [],
        'u_o': [],
        'os_shared' : [],
        'sa_shared' : [],
        'oa_shared' : [],
        'osa_shared' : [],
        'o_total': [],
        'a_total' : [],
        's_total' : [],
        'all_parts': []
    }

vars = values_rs.keys()
for t in range(len(eeg_data['times'])):
     for var in vars:
          values_rs[var].append(results[t][0][var])
          ps_rs[var].append(results[t][1][var])

results_rs = (values_rs, ps_rs)

results_dict = {
    'data_split': args.data_split, 
    'distance_type': args.distance_type,
	'results': results_rs,
	'times': eeg_data['times'],
    'sfreq': args.sfreq
}

res_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/{args.data_split}/vp-results/z_{args.zscore}/sfreq-{format(args.sfreq, "04")}/GA/'
if os.path.isdir(res_folder) == False:
	os.makedirs(res_folder)

file_path = res_folder + f'vp_results.pkl'

# Save all model rdms
with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)

