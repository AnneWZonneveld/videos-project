"""
Calculating the similarity between object model RDM based using the 'most frequent label'  
and EEG RDMs for all time points for subject level.

(to enable comparison to using object model RDM using the average embedding)

Code outdated.

Parameters
----------
sub: int
    Subject nr
distance_type: str
    Whether to use EEG RDMs based on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
bin_width: int
    Bin width used for EEG smoothening

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
parser.add_argument('--bin_width', default=0, type=float) 
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


######################### Temporal smoothening ####################################

if not args.bin_width == 0:
    og_bin_width = abs(eeg_data['times'][0] - eeg_data['times'][1])
    smooth_factor = int(round((args.bin_width / og_bin_width), 1))
    # smooth_factor = round((bin_width / og_bin_width), 1)

    n_samples = int(eeg_rdms.shape[2]/smooth_factor)
    smooth_eeg_rdms = np.zeros((eeg_rdms.shape[0], eeg_rdms.shape[1], n_samples))
    t_samples = []
    for i in range(n_samples):
        id_1 = int(i * smooth_factor)
        id_2 = int(id_1 + smooth_factor)
        smooth_eeg_rdms[:, :, i] = np.mean(eeg_rdms[:,:,id_1:id_2], axis=2)
        t_samples.append(eeg_data['times'][id_1])

    eeg_data['times'] = t_samples
    eeg_rdms = smooth_eeg_rdms

############################# Analysis ###########################################

# Calculate R^2
print('Starting multiprocessing R^2')
results = calc_rsquared_mp(eeg_rdms=eeg_rdms, feature_rdm=feature_rdm, jobarr_id=args.jobarr_id, its=10000, n_cpus=n_cpus, shm_name='freq')
print('Done multiprocessing')

############################ Save data #############################################

results_dict = {
    'data_split': data_split, 
    'sub': args.sub, 
    'distance_type': args.distance_type,
    'bin_width': args.bin_width,
	'cor_values': results,
	'times': eeg_data['times']
}

res_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/standard/sub-{sub_format}/{args.distance_type}/bin_{args.bin_width}/' 
if not os.path.exists(res_folder) == True:
    os.mkdir(res_folder)

file_path = res_folder + f'cors_objects_freq.pkl'

# Save all model rdms
with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)


