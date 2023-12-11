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

data_split = 'train'
n_cpus = 4

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
model_file = '/scratch/azonneveld/rsa/model/rdms/t2/rdm_t2_GA_objects.pkl' 
with open(model_file, 'rb') as f: 
    model_rdms = pickle.load(f)
feature_rdm = model_rdms['GA_rdm']

# Neural RDMs
print('Loading neural rdms')
sub_format = format(args.sub, '02')
eeg_file = f'/scratch/azonneveld/rsa/eeg/rdms/sub-{sub_format}/avg_{args.distance_type}.pkl'

with open(eeg_file, 'rb') as f: 
    eeg_data = pickle.load(f)
eeg_rdms = eeg_data['rdms_array']

############################# Analysis ###########################################

# Calculate CI
print('Starting multiprocessing CI')
results = calc_cis_mp(eeg_rdms=eeg_rdms, feature_rdm=feature_rdm, jobarr_id=args.jobarr_id, its=1000, n_cpus=n_cpus, shm_name='freq')
print('Done multiprocessing')

############################ Save data #############################################

results_dict = {
    'data_split': data_split, 
    'sub': args.sub,
    'distance_type': args.distance_type,
	'cis_values': results,
	'times': eeg_data['times']
}

res_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/sub-{sub_format}/' 
if not os.path.exists(res_folder) == True:
    os.mkdir(res_folder)

file_path = res_folder + f'cis_objects_freq.pkl'

# Save all model rdms
with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)
