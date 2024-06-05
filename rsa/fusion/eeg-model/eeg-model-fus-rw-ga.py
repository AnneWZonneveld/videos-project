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
n_cpus = 2

parser = argparse.ArgumentParser()
parser.add_argument('--zscore', default=0, type=int)
parser.add_argument('--features', default='o-s-a', type=str)
parser.add_argument('--distance_type', default='euclidean-cv', type=str) 
parser.add_argument('--jobarr_id', default=1, type=int) 
parser.add_argument('--bin_width', default=0, type=float)
parser.add_argument('--slide', default=0, type=int)
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
subs = 3
eeg_rdms_subs = np.zeros((subs, 1000, 1000, 185))
for i in range(subs):
    sub_format = format(i+1, '02')
    eeg_file = f'/scratch/azonneveld/rsa/eeg/rdms/z_{args.zscore}/sub-{sub_format}/avg_{args.distance_type}_{args.slide}.pkl'
    with open(eeg_file, 'rb') as f: 
        eeg_data = pickle.load(f)
    eeg_rdms = eeg_data['rdms_array']
    eeg_rdms_subs[i, :, :, :] = eeg_rdms

GA_rdms = np.mean(eeg_rdms_subs, axis=0)


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

    eeg_rdms = smooth_eeg_rdms
    eeg_data['times'] = t_samples

######################## Reweighting of RDMs #######################################
vars_fm = args.features.split('-')

vars = []
sq_model_rdms = {}
for var_fm in vars_fm:
    if var_fm == 'o':
        var = 'objects'
    elif var_fm == 's':
        var = 'scenes'
    elif var_fm == 'a':
        var = 'actions'

    vars.append(var)

    sq_rdm = squareform(model_rdms[var], checks=False)
    sq_model_rdms[var] = sq_rdm[:, np.newaxis]

design_matrix = np.ones(sq_model_rdms[var].shape[0])[:, np.newaxis]
for var in vars:
    design_matrix = np.concatenate((sq_model_rdms[var], design_matrix), axis=1)

# Calculate R^2
print('Starting multiprocessing reweighted R^2')
results = calc_rsquared_rw_mp(eeg_rdms=GA_rdms, design_matrix=design_matrix, jobarr_id=args.jobarr_id, n_cpus=n_cpus, shm_name=f'rw_ga_{args.features}')
print('Done multiprocessing')

###################### Save ###############################

results_dict = {
    'data_split': data_split, 
    'features': args.features,
    'distance_type': args.distance_type,
	'cor_values': results,
	'times': eeg_data['times'],
    'bin_width': args.bin_width
}

res_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/reweighted/z_{args.zscore}/bin_{args.bin_width}/slide_{args.slide}/GA/' 
if not os.path.exists(res_folder) == True:
    os.mkdir(res_folder)

file_path = res_folder + f'cors_{args.features}.pkl'

# Save all model rdms
with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)


