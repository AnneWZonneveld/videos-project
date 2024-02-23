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
from functools import partial
import multiprocessing as mp
from vp_utils import *
from vp_multiprocessing import *

###################### Set parameters ###########################################
n_cpus = 16

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--zscore', default=0, type=int)
parser.add_argument('--feature', default='objects', type=str)
parser.add_argument('--distance_type', default='euclidean-cv', type=str) 
parser.add_argument('--eval_method', default='spearman', type=str)
parser.add_argument('--jobarr_id', default=1, type=int) 
parser.add_argument('--sfreq', default=500, type=int)
parser.add_argument('--model_metric', default='euclidean', type=str)

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
feature_rdm = model_rdms[args.feature]

# Neural RDMs
print('Loading neural rdms')
sub_format = format(args.sub, '02')
sfreq_format = format(args.sfreq, '04')
eeg_file = f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_{args.zscore}/sub-{sub_format}/avg_{args.distance_type}_{sfreq_format}.pkl'

with open(eeg_file, 'rb') as f: 
    eeg_data = pickle.load(f)
eeg_rdms = eeg_data['rdms_array']

############################# Analysis ###########################################
    
print('Starting multiprocessing R^2')
shm_name = f'shm_{args.sub}_{args.data_split}_{args.distance_type}_{args.sfreq}_{args.eval_method}'
results = calc_rsquared_mp(eeg_rdms=eeg_rdms, feature_rdm=feature_rdm, jobarr_id=args.jobarr_id, its=10000, n_cpus=n_cpus, shm_name=shm_name, eval_method=args.eval_method)
print('Done multiprocessing')

############################ Save data #############################################

results_dict = {
    'data_split': args.data_split, 
    'zscore': args.zscore,
    'sub': args.sub, 
    'feature': args.feature,
    'distance_type': args.distance_type,
    'eval_method': args.eval_method,
	'cor_values': results,
	'times': eeg_data['times'],
    'sfreq' : args.sfreq
}

res_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/model_{args.model_metric}/{args.data_split}/standard/z_{args.zscore}/sub-{sub_format}/{args.distance_type}/sfreq_{sfreq_format}/{args.eval_method}/' 
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)

file_path = res_folder + f'cors_{args.feature}.pkl'

# Save all model rdms
with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)


