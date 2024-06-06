"""
Calculating the 95% confidence interval between specified model RDM and fMRI RDM on subject level

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
parser.add_argument('--sub', type=int)
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
sub_format = format(args.sub, '02')

####################### Load model RDM & neural data ###############################
print('Loading model rdms')
model_file = f'/scratch/azonneveld/rsa/model/rdms/t2/{args.model_metric}/rdm_t2_avg.pkl' 
with open(model_file, 'rb') as f: 
    model_rdms = pickle.load(f)
model_rdm = model_rdms[args.data_split][args.feature]

fmri_file = f'/scratch/azonneveld/rsa/fmri/rdms/sub-{sub_format}/{args.distance_type}/{args.data_split}_rdms.pkl'
with open(fmri_file, 'rb') as f: 
    data = pickle.load(f)
fmri_rdms = data['results']

###################### Analysis ###################################################

print("Start multiprocessing cis calc")
results = calc_cis_mp(rois=rois, fmri_data=fmri_rdms, feature_rdm=model_rdm, its=1000, n_cpus=args.n_cpus, eval_method=args.eval_method)
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

res_folder = f'/scratch/azonneveld/rsa/fusion/fmri-model/standard/{args.data_split}/model_{args.model_metric}/sub-{sub_format}/{args.distance_type}/{args.eval_method}/'
if os.path.isdir(res_folder) == False:
	os.makedirs(res_folder)
res_file = res_folder + f'cis_{args.feature}.pkl'

with open(res_file, 'wb') as f:
    pickle.dump(results_dict, f)

