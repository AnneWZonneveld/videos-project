"""
Mean distance RDM over time for group level. 

Parameters
----------
zscore : int
	Whether to z-score [1] or not [0] the data.
data_split: str
    Train or test. 
distance_type: str
    Whether to base the RDMs on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
sfeq: int
    Sampling frequency. 

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
import tensorflow as tf
import tensorflow_hub as hub
from scipy.stats import pearsonr
from matplotlib.pyplot import cm
import argparse

# Input argurments
parser = argparse.ArgumentParser()
parser.add_argument('--distance_type', default='euclidean', type=str)
parser.add_argument('--zscore', default=0, type=int)
parser.add_argument('--sfreq', default=500, type=int)
parser.add_argument('--data_split', default='train', type=str)
args = parser.parse_args()

# Parameters
res_folder = '/scratch/azonneveld/rsa/eeg/plots/dist-eval/'
distance_type = args.distance_type
n_subs = 6
sfreq_format =  format(args.sfreq, '04')

# Load in RDMs for all subjects
# subs_data = np.zeros((n_subs, 1000, 1000, 185))
subs_data = np.zeros((n_subs, 499500, 185))
for i in range(n_subs):
    sub = i + 1
    sub_format = format(sub, '02')

    with open(f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_{args.zscore}/sub-{sub_format}/avg_{distance_type}_{sfreq_format}.pkl', 'rb') as f:
        data = pickle.load(f)
    
    rdms = data['rdms_array']
    times = data['times']
    del data

    for t in range(subs_data.shape[2]):
        squareform_rdm = squareform(rdms[:,:,t])
        subs_data[i, :, t] = squareform_rdm
    
    sub_data = subs_data[i, :, :]
    mean_distance = np.mean(sub_data, axis=0)
    print(mean_distance.shape)
    sub_df = pd.DataFrame()
    sub_df['distance'] = mean_distance
    sub_df['time'] = times

    # Create individual plot
    fig, ax = plt.subplots(1,1)
    sns.lineplot(data=sub_df, x='time', y='distance')
    ax.set_title(f'sub {sub} {distance_type}', fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel(f"Average over RDM", fontsize=10)
    fig.tight_layout()
    img_path = res_folder + f'sub_{sub}_{distance_type}_z{args.zscore}_{sfreq_format}.png'
    plt.savefig(img_path)
    plt.clf()

# Put averaged RDM data in long format
subs_df = pd.DataFrame()
for i in range(n_subs):
    sub = i + 1
    sub_format = format(sub, '02')

    sub_data = subs_data[i, :, :]
    sub_avg = np.mean(sub_data, axis=0)

    sub_df = pd.DataFrame()
    sub_df['distance'] = sub_avg
    sub_df['time'] = times
    sub_df['sub'] = len(sub_avg) * [sub]

    subs_df = pd.concat([subs_df, sub_df], axis=0)

subs_df = subs_df.reset_index()

# Create plot
fig, ax = plt.subplots(1,1)
sns.lineplot(data=subs_df, x='time', y='distance')
ax.set_title(f'{distance_type}', fontsize=12)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_ylabel(f"Average over RDM", fontsize=10)
fig.tight_layout()
img_path = res_folder + f'{distance_type}_z{args.zscore}_{sfreq_format}.png'
plt.savefig(img_path)
plt.clf()



