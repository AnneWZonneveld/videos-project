"""
Descriptives of specified RDM on subject-level after having computed the RDM. 
- Mean distance over time
- Rank over time
- Rank ratio over time (ratio between rank and nr of rdm entries (1 would be full rank))

Parameters
----------
sub : int
	Used subject.
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
from scipy.stats import pearsonr
from matplotlib.pyplot import cm
import argparse

# Input argurments
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--data_split', default='train', type=str) 
parser.add_argument('--distance_type', default='euclidean', type=str) 
parser.add_argument('--sfreq', default=500, type=int)
args = parser.parse_args()

sub_format =  format(args.sub, '02')
sfreq_format =  format(args.sfreq, '04')

res_folder = f'/scratch/azonneveld/rsa/eeg/plots/dist-eval/{args.data_split}/{sub_format}/{args.distance_type}/'
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)

# Load data
with open(f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_{args.zscore}/sub-{sub_format}/avg_{args.distance_type}_{sfreq_format}.pkl', 'rb') as f:
    data = pickle.load(f)

rdms = data['rdms_array']
times = data['times']
sample_ts = [5, 20, 60, 100, 140]

mean_distances = []
ranks = []
ranks_ratios = []
for t in range(len(times)):
    rdm_t = rdms[:, :, t]

    mean_dist = np.mean(rdm_t)
    mean_distances.append(mean_dist)

    rank = np.linalg.matrix_rank(rdm_t)
    rank_ratio = rank / rdm_t.shape[0]
    ranks.append(rank)
    ranks_ratios.append(rank_ratio)

    if t in sample_ts:

        fig, axes = plt.subplots(dpi=300)
        sns.displot(squareform(rdm_t))

        fig.tight_layout()

        img_path = res_folder + f'rdm_dist_t{t}.png'
        plt.savefig(img_path)
        plt.clf()


# Mean distance over time plot
sub_df = pd.DataFrame()
sub_df['distance'] = mean_distances
sub_df['time'] = times        

fig, ax = plt.subplots(1,1)
sns.lineplot(data=sub_df, x='time', y='distance')
ax.set_title(f'sub {sub_format} {args.distance_type}', fontsize=12)
# ax.set_title(f'sub {sub_format} pearson', fontsize=12)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_ylabel(f"Average over RDM", fontsize=10)
fig.tight_layout()
img_path = res_folder + f'mean_distance.png'
plt.savefig(img_path)
plt.clf()

# Plot rank
sub_df = pd.DataFrame()
sub_df['rank'] = ranks
sub_df['time'] = times        

fig, ax = plt.subplots(1,1)
sns.lineplot(data=sub_df, x='time', y='rank')
# ax.set_title(f'sub {sub_format} {args.distance_type}', fontsize=12)
ax.set_title(f'sub {sub_format} pearson', fontsize=12)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_ylabel(f"Absolute rank", fontsize=10)
fig.tight_layout()
img_path = res_folder + f'rank.png'
plt.savefig(img_path)
plt.clf()

# Plot rank ratio
sub_df = pd.DataFrame()
sub_df['rank_ratio'] = ranks_ratios
sub_df['time'] = times        

fig, ax = plt.subplots(1,1)
sns.lineplot(data=sub_df, x='time', y='rank_ratio')
# ax.set_title(f'sub {sub_format} {args.distance_type}', fontsize=12)
ax.set_title(f'sub {sub_format} pearson', fontsize=12)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_ylabel(f"Rank ratio", fontsize=10)
fig.tight_layout()
img_path = res_folder + f'rank_ratio.png'
plt.savefig(img_path)
plt.clf()