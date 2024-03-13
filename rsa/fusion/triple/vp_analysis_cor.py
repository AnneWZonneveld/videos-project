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
import argparse
import pingouin as pg
from vp_multiprocessing import *


###################### Set parameters ###########################################
parser = argparse.ArgumentParser()
parser.add_argument('--data_split', default='training', type=str)
parser.add_argument('--roi', type=str)
parser.add_argument('--eeg_distance_type', default='euclidean-cv', type=str) 
parser.add_argument('--fmri_distance_type', default='euclidean-cv', type=str) 
parser.add_argument('--model_metric', default='correlation', type=str)
parser.add_argument('--eval_method', default='spearman', type=str)
parser.add_argument('--feature_oi', type=str) 
parser.add_argument('--jobarr_id', default=1, type=int) 
parser.add_argument('--n_cpus', default=1, type=int) 
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

########################### Loading in data #####################################

# Model RDMs
print('Loading model rdms')
model_file = f'/scratch/azonneveld/rsa/model/rdms/t2/{args.model_metric}/rdm_t2_avg.pkl' 
with open(model_file, 'rb') as f: 
    model_rdms = pickle.load(f)
model_rdms = model_rdms[args.data_split]

# fMRI RDMs
print('Loading fMRI rdms')
fmri_file = f'/scratch/azonneveld/rsa/fmri/rdms/GA/{args.fmri_distance_type}/GA_{args.data_split}_rdm.pkl'

with open(fmri_file, 'rb') as f: 
    fmri_data = pickle.load(f)
fmri_rdms = fmri_data['rdm_array']
roi_rdm = fmri_rdms[args.roi]

# EEG RDMs
print('Loading EEG rdms')
eeg_file = f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_1/GA/avg_{args.eeg_distance_type}_0500.pkl'

with open(eeg_file, 'rb') as f: 
    eeg_data = pickle.load(f)
eeg_rdms = eeg_data['rdms_array']


################################ Analysis #########################
print('Starting multiprocessing reweighted R^2')
shm_name = f'vp_{args.data_split}_{args.feature_oi}_{args.roi}'
results = calc_common_mp(eeg_rdms=eeg_rdms, roi_rdm=roi_rdm, model_rdms=model_rdms, feature_oi=args.feature_oi, jobarr_id=args.jobarr_id, its=10000, n_cpus=args.n_cpus, shm_name=shm_name, method=args.eval_method)
print('Done multiprocessing')

################################ Save #############################
values = []
ps = []
for t in range(len(eeg_data['times'])):
    values.append(results[t][0])
    ps.append(results[t][1])
results_rs = (values, ps)

results_dict = {
    'data_split': args.data_split,
    'fmri_distance_type': args.fmri_distance_type,
    'eeg_distance_type': args.eeg_distance_type,
	'results': results_rs,
	'times': eeg_data['times'],
    'feature_oi': args.feature_oi,
    'roi': args.roi,
    'eval_method': args.eval_method
}

res_folder = f'/scratch/azonneveld/rsa/fusion/triple/{args.data_split}/{args.roi}/{args.feature_oi}/fmri_{args.fmri_distance_type}/eeg_{args.eeg_distance_type}/'
if os.path.isdir(res_folder) == False:
	os.makedirs(res_folder)

file_path = res_folder + f'vp_results_{args.eval_method}.pkl'

# Save all model rdms
with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)
