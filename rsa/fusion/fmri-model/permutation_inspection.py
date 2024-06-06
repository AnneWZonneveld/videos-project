"""
Performing variance partioning between model RDMs and fMRI RDM.
Specifically to inspect the null distribution (i.e. plotting).
Group level. 

Parameters
----------
data_split: str
    Train or test. 
distance_type: str
    Whether to use EEG RDMs based on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
rois: str
    String with all ROIs.
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
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import argparse
from fmri_model_mp import *

parser = argparse.ArgumentParser()
parser.add_argument('--distance_type', default='pearson', type=str) 
parser.add_argument('--data_split', default='training', type=str)
parser.add_argument('--rois', type=str)
parser.add_argument('--eval_method', default='spearmean', type=str)
parser.add_argument('--model_metric', default='euclidean', type=str)
parser.add_argument('--n_cpus', default=1, type=int)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
     
rois = args.rois.split(',') 

########################### Loading in data #####################################

# Model RDMs
print('Loading model rdms')
model_file = f'/scratch/azonneveld/rsa/model/rdms/t2/{args.model_metric}/rdm_t2_avg.pkl' #change this to permuted freq matrix? / avg
with open(model_file, 'rb') as f: 
    model_rdms = pickle.load(f)
model_rdms = model_rdms[args.data_split]

# Neural RDMs
print('Loading neural rdms')
fmri_file = f'/scratch/azonneveld/rsa/fmri/rdms/GA/{args.distance_type}/GA_{args.data_split}_rdm.pkl'

with open(fmri_file, 'rb') as f: 
    fmri_data = pickle.load(f)
fmri_rdms = fmri_data['rdm_array']

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

###################### Analysis ###################################################

print("Start multiprocessing correlation calc")
print(f"ROIs main {rois}")
results =results = calc_rsquared_rw_mp(rois=rois, fmri_data=fmri_rdms, design_matrices=design_matrices, n_cpus=args.n_cpus, its=100000, inspect=True)
print("Done multiprocessing")
