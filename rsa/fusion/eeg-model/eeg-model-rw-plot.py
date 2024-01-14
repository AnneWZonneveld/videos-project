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

# Take arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--distance_type', default='euclidean-cv', type=str)
parser.add_argument('--bin_width', default=0, type=float)
parser.add_argument('--zscore', default=0, type=int)
parser.add_argument('--slide', default=0, type=int)
parser.add_argument('--cv_r2', default=0, type=int)
parser.add_argument('--ridge', default=0, type=int)
args = parser.parse_args()


####################### Load data #######################################
sub_format = format(args.sub, '02')
if args.bin_width == 0:
    bin_format = 0.0
else:
    bin_format = args.bin_width

# sub_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/reweighted/z_{args.zscore}/bin_{args.bin_width}/slide_{args.slide}/sub-{sub_format}/' 
# res_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/reweighted/z_{args.zscore}/bin_{args.bin_width}/slide_{args.slide}/plots/'
sub_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/reweighted/z_{args.zscore}/bin_{bin_format}/slide_{args.slide}/sub-{sub_format}/ridge_{args.ridge}/cv_{args.cv_r2}/' 
res_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/reweighted/plots/z_{args.zscore}/bin_{bin_format}/slide_{args.slide}/sub-{sub_format}/ridge_{args.ridge}/cv_{args.cv_r2}/'
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)

feature_combs = ['o-s-a', 'o-s', 'o-a', 's-a', 'o', 's', 'a']

cors = {}
for comb in feature_combs:
    cors[comb] = []

for comb in feature_combs:

    cor_path = sub_folder + f'cors_{comb}.pkl'
    with open(cor_path, 'rb') as f: 
        cor_res = pickle.load(f)
    cor_values = cor_res['cor_values']

    for i in range(len(cor_values)):
        cors[comb].append(cor_values[i][0])

######################### Unique variance #####################################
u_a_cors = (np.array(cors['o-s-a']) - np.array(cors['o-s']))*100
u_s_cors = (np.array(cors['o-s-a']) -  np.array(cors['o-a']))*100
u_o_cors = (np.array(cors['o-s-a']) -  np.array(cors['s-a']))*100
osa_cors = np.array(cors['o-s-a']) * 100

features_cor_dict = {
    'objects': u_o_cors,
    'scenes': u_s_cors,
    'actions': u_a_cors
}


################## Shared variance (Pablo suggestion) #######################################
os_shared = osa_cors - np.array(cors['a'])*100 - u_o_cors - u_s_cors
sa_shared = osa_cors - np.array(cors['o'])*100 - u_s_cors - u_a_cors
oa_shared = osa_cors - np.array(cors['s'])*100 - u_o_cors - u_a_cors

osa_shared = osa_cors - u_o_cors - u_s_cors - u_a_cors - os_shared - sa_shared - oa_shared

shared_cor_dict = {
    'o-s': os_shared,
    's-a': sa_shared,
    'o-a' : oa_shared,
    'o-s-a': osa_shared
}


##################### Plot: Standard variance explained #############################
print('Plotting standard variance explained')

features = ['o', 's', 'a']
colours = ['b', 'r', 'g']

fig, ax = plt.subplots(dpi=300)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    if feature == 'o':
        feature_lab = 'objects'
    elif feature == 's':
        feature_lab = 'scenes'
    elif feature == 'a':
        feature_lab = 'actions'

    stats_df = pd.DataFrame()
    stats_df['cors'] = np.array(cors[feature])*100
    stats_df['times'] = cor_res['times']
    ax.plot(stats_df['times'], stats_df['cors'], label=feature_lab, color=colour)

ax.plot(stats_df['times'], osa_cors, label='full model', color='gray', alpha=0.8)
ax.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
ax.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
ax.set_title(f'EEG-model {sub_format}, ridge={args.ridge}, cv={args.cv_r2}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Variance exlained (%)')
ax.legend()
fig.tight_layout()

img_path = res_folder + f'{sub_format}_r2.png'
plt.savefig(img_path)
plt.clf()


######################### Plot: unique variance ####################################
print('Plotting unique variance explained')

features = ['objects', 'scenes', 'actions']

colours = ['b', 'r', 'g']
fig, ax = plt.subplots(dpi=300)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    stats_df = pd.DataFrame()
    stats_df['cors'] = features_cor_dict[feature]
    stats_df['times'] = cor_res['times']
    ax.plot(stats_df['times'], stats_df['cors'], label=feature, color=colour)

ax.plot(stats_df['times'], osa_cors, label='full model', color='gray', alpha=0.8)
ax.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
ax.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
ax.set_title(f'EEG-model {sub_format}, ridge={args.ridge}, cv={args.cv_r2}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Unique variance exlained (%)')
ax.legend()
fig.tight_layout()

img_path = res_folder + f'{sub_format}_unique_r2.png'
plt.savefig(img_path)
plt.clf()

######################### Plot: shared variance variance ####################################
print('Plotting shared variance')

combs = ['o-s', 'o-a', 's-a','o-s-a']

colours = ['b', 'r', 'g', 'orange']
fig, ax = plt.subplots(dpi=300)
for i in range(len(combs)):
    comb = combs [i]
    colour = colours[i]

    stats_df = pd.DataFrame()
    stats_df['cors'] = shared_cor_dict[comb]
    stats_df['times'] = cor_res['times']
    ax.plot(stats_df['times'], stats_df['cors'], label=comb, color=colour)

# ax.plot(stats_df['times'], osa_cors, label='full model', color='gray', alpha=0.8)
ax.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
ax.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
ax.set_title(f'EEG-model sub {sub_format}, ridge={args.ridge}, cv={args.cv_r2}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Shared variance explained (%)')
ax.legend()
fig.tight_layout()

img_path = res_folder + f'{sub_format}_shared_r2.png'
plt.savefig(img_path)
plt.clf()