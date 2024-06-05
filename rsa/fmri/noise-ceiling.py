"""
Calculating noise ceiling per ROI.

Parameters
----------
data_split: str
    Train or test. 
distance_type: str
    Whether to base the RDMs on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
roi: str
    ROI.

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

parser = argparse.ArgumentParser()
parser.add_argument('--distance_type', default='pearson', type=str) 
parser.add_argument('--data_split', default='training', type=str)
parser.add_argument('--roi', type=str)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
     
####################### Load neural data ###############################
n_subs = 10

if args.roi == 'RSC':
     n_subs = 9
elif args.roi == 'TOS':
     n_subs = 8
     
if args.data_split == 'test':
     n_conditions = 102
elif args.data_split == 'train':
     n_conditions = 1000

subs_array = np.zeros((n_subs, n_conditions, n_conditions))
for i in range(n_subs):
    sub = i + 1
    sub_format = format(sub, '02')

    with open(f'/scratch/azonneveld/rsa/fmri/rdms/sub-{sub_format}/{args.distance_type}/{args.data_split}_rdms.pkl', 'rb') as f:
        data = pickle.load(f)
    
    rdms = data['results']

    try:
        roi_rdm = rdms[args.roi]
    except:
         print(f'sub {sub} {args.roi} not present')

    subs_array[i, :, :] = roi_rdm
   
    del data, rdms

with open(f'/scratch/azonneveld/rsa/fmri/rdms/GA/{args.distance_type}/GA_{args.data_split}_rdm.pkl', 'rb') as f:
    GA_rdms = pickle.load(f) 
GA_rdms = GA_rdms['rdm_array']
GA_rdm = GA_rdms[args.roi]
del GA_rdms 

#################### Calc noise ceiling ################################
noiseLower = np.zeros(n_subs)
noiseHigher = np.zeros(n_subs)

for sub in range(n_subs):
    
    sub_rdm = subs_array[sub, :, :]
    noiseHigher[sub]= spearmanr(squareform(sub_rdm.round(5)), squareform(GA_rdm.round(5)))[0]
    
    mask = np.ones(n_subs,dtype=bool)
    mask[sub] = 0
    rdms_without = subs_array[mask, :, :]
    GA_without = np.mean(rdms_without, axis=0)
    noiseLower[sub] = spearmanr(squareform(sub_rdm.round(5)), squareform(GA_without.round(5)))[0]

noiseCeiling = {}
noiseCeiling['UpperBound'] = np.mean(noiseHigher, axis=0)
noiseCeiling['LowerBound'] = np.mean(noiseLower, axis=0)

results_dict = {
    'data_split': args.data_split, 
    'distance_type': args.distance_type,
    'roi': args.roi,
    'results': noiseCeiling
}

res_folder = f'/scratch/azonneveld/rsa/fmri/noise_ceiling/{args.data_split}/{args.distance_type}/{args.roi}/'
if os.path.isdir(res_folder) == False:
	os.makedirs(res_folder)
res_file = res_folder + f'noise_ceiling.pkl'

with open(res_file, 'wb') as f:
    pickle.dump(results_dict, f)
