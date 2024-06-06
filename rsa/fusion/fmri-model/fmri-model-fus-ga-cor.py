"""
Calculating similarity between specified model RDM and fMRI RDM. 
Group level --> including normalization using noise ceiling. First perform rsa/fmri/noise-ceiling.py

Parameters
----------
sub: int
    Subject nr
feature: str
    'Objects', 'scenes', 'actions' model.
data_split: str
    Train or test. 
distance_type: str
    Whether to use EEG RDMs based on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
rois: str
    String with all ROIs.
eval_method: str
    Method to compute similarity between RDMS; 'spearman' or 'pearson'
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
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import argparse
from fmri_model_mp import *


parser = argparse.ArgumentParser()
parser.add_argument('--feature', type=str)
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

####################### Load model RDM & neural data & noice ceilings ###############################
print('Loading model rdms')
model_file = f'/scratch/azonneveld/rsa/model/rdms/t2/{args.model_metric}/rdm_t2_avg.pkl' 
with open(model_file, 'rb') as f: 
    model_rdms = pickle.load(f)
model_rdm = model_rdms[args.data_split][args.feature]

print ('Loading fmri rdms')
fmri_file = f'/scratch/azonneveld/rsa/fmri/rdms/GA/{args.distance_type}/GA_{args.data_split}_rdm.pkl'
with open(fmri_file, 'rb') as f: 
    data = pickle.load(f)
fmri_rdms = data['rdm_array']

print("Loading noise ceilings")
noise_ceilings = {}
for i in range(len(rois)):
    roi = rois[i]
    
    nc_file = f'/scratch/azonneveld/rsa/fmri/noise_ceiling/{args.data_split}/{args.distance_type}/{roi}/noise_ceiling.pkl'
    with open(nc_file, 'rb') as f: 
        data = pickle.load(f)
    nc = data['results']
    noise_ceilings[roi] = nc

###################### Analysis ###################################################

print("Start multiprocessing correlation calc")
results = calc_cor_ga_mp(rois=rois, fmri_data=fmri_rdms, feature_rdm=model_rdm, its=10000, n_cpus=args.n_cpus, eval_method=args.eval_method, noise_ceilings=noise_ceilings)
print("Done multiprocessing")

rois_dict = {}
for i in range(len(rois)):
    roi = rois[i]
    rois_dict[roi] = results[i]

############################ Save results ############################
results_dict = {
    'feature': args.feature,
    'data_split': args.data_split, 
    'distance_type': args.distance_type,
    'eval_method': args.eval_method,
    'model_metric': args.model_metric,
    'results': rois_dict
}

res_folder = f'/scratch/azonneveld/rsa/fusion/fmri-model/standard/{args.data_split}/model_{args.model_metric}/GA/{args.distance_type}/{args.eval_method}/'
if os.path.isdir(res_folder) == False:
	os.makedirs(res_folder)
res_file = res_folder + f'cors_{args.feature}.pkl'

with open(res_file, 'wb') as f:
    pickle.dump(results_dict, f)

