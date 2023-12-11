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
from functools import partial
import multiprocessing as mp
from vp_utils import *
from vp_multiprocessing import *

###################### Set parameters ###########################################

data_split = 'train'
n_cpus = 8

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--distance_type', default='euclidean-cv', type=str) 
parser.add_argument('--jobarr_id', default=1, type=int) 
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


########################### Loading in data #####################################

# Model RDMs
print('Loading model rdms')
model_file = '/scratch/azonneveld/rsa/model/rdms/t2/rdm_t2_avg.pkl' #change this to permuted freq matrix? / avg
with open(model_file, 'rb') as f: 
    model_rdms = pickle.load(f)
model_rdms = model_rdms[data_split]

# Neural RDMs
print('Loading neural rdms')
sub_format = format(args.sub, '02')
eeg_file = f'/scratch/azonneveld/rsa/eeg/rdms/sub-{sub_format}/avg_{args.distance_type}.pkl'

with open(eeg_file, 'rb') as f: 
    eeg_data = pickle.load(f)
eeg_rdms = eeg_data['rdms_array']


######################## Reweighting of RDMs #######################################
vars = ['objects', 'scenes', 'actions'] # --> should be specified in job array 

sq_model_rdms = {}
for var in vars:
    sq_rdm = squareform(model_rdms[var], checks=False)
    sq_model_rdms[var] = sq_rdm[:, np.newaxis]

design_matrix = np.ones(sq_model_rdms[var].shape[0])[:, np.newaxis]
for var in vars:
    design_matrix = np.concatenate((sq_model_rdms[var], design_matrix), axis=1)

times = eeg_data['times'].shape[0]
for t in range(len(times)):
    eeg_rdm = eeg_rdms[:, :, t]
    sq_eeg_rdm  = squareform(eeg_rdm)

    model = LinearRegression().fit(design_matrix, sq_eeg_rdm)
    coefs = model.coef_

    rw_rdms = 0
    for i in range(len(vars)):
        var = vars[i]
        rw_rdm = coefs[i] * sq_model_rdms[var]
        added_rdms = added_rdms + rw_rdm
    rw_rdms = added_rdms + coefs[-1]

    # Calculate correlation & p
    cor = spearmanr(rw_rdms, sq_eeg_rdm)[0]

    # Calculate correlational variability 


