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
parser.add_argument('--distance_type', default='euclidean-cv', type=str)
parser.add_argument('--bin_width', default=0, type=float)
parser.add_argument('--zscore', default=0, type=int)
parser.add_argument('--slide', default=0, type=int)
parser.add_argument('--ridge', default=0, type=int)
parser.add_argument('--cv_r2', default=0, type=int)
args = parser.parse_args()


####################### Load data #######################################
sub_format = sub_format = format(args.sub, '02')
if args.bin_width == 0:
    bin_format = 0.0

sub_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/reweighted/z_{args.zscore}/bin_{bin_format}/slide_{args.slide}/sub-{sub_format}/ridge_{args.ridge}/cv_{args.cv_r2}/' 
res_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/reweighted/negative-var-analysis/'
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

############################ Plot  1 ######################################
fig, ax = plt.subplots(1,3, dpi=300, figsize=(30,10))

# Plot 1: objects
ax[0].set_title(f'objects')
ax[0].plot(cor_res['times'], np.array(cors['o'])*100, label='objects')
ax[0].plot(cor_res['times'], u_o_cors, label='unique')
ax[0].plot(cor_res['times'], os_shared, label='shared scenes')
ax[0].plot(cor_res['times'], oa_shared, label='shared actions')
ax[0].plot(cor_res['times'], osa_shared, label='shared scenes + actions')
control = u_o_cors + os_shared + oa_shared + osa_shared
ax[0].plot(cor_res['times'], control, label='control', color='gray', linestyle='dotted')
ax[0].axvline(x=0, color='gray', alpha=0.5, linestyle='--')
ax[0].axvline(x=3, color='gray', alpha=0.5, linestyle='--')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Unique variance exlained (%)')
ax[0].legend()

# Plot 2: scenes
ax[1].set_title(f'scenes')
ax[1].plot(cor_res['times'], np.array(cors['s'])*100,  label='scenes')
ax[1].plot(cor_res['times'], u_s_cors, label='unique')
ax[1].plot(cor_res['times'], os_shared, label='shared objects')
ax[1].plot(cor_res['times'], sa_shared, label='shared actions')
ax[1].plot(cor_res['times'], osa_shared, label='shared object + actions')
control = u_s_cors + os_shared + sa_shared + osa_shared
ax[1].plot(cor_res['times'], control, label='control', color='gray', linestyle='dotted')
ax[1].axvline(x=0, color='gray', alpha=0.5, linestyle='--')
ax[1].axvline(x=3, color='gray', alpha=0.5, linestyle='--')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('')
ax[1].legend()

# Plot 3: actions
ax[2].set_title(f' actions')
ax[2].plot(cor_res['times'], np.array(cors['a'])*100, label='actions')
ax[2].plot(cor_res['times'], u_a_cors, label='unique')
ax[2].plot(cor_res['times'], oa_shared, label='shared objects')
ax[2].plot(cor_res['times'], sa_shared, label='shared scenes')
ax[2].plot(cor_res['times'], osa_shared, label='shared object + actions')
control = u_a_cors + oa_shared + sa_shared + osa_shared
ax[2].plot(cor_res['times'], control, label='control', color='gray', linestyle='dotted')
ax[2].axvline(x=0, color='gray', alpha=0.5, linestyle='--')
ax[2].axvline(x=3, color='gray', alpha=0.5, linestyle='--')
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('')
ax[2].legend()

fig.tight_layout()
img_path = res_folder + f'sub_{sub_format}.png'
plt.savefig(img_path)
plt.clf()

############################ Plot  2 ######################################
fig, ax = plt.subplots(dpi=300)
ax.plot(cor_res['times'], np.array(cors['o'])*100, label='o')
ax.plot(cor_res['times'], np.array(cors['s'])*100, label='s')
ax.plot(cor_res['times'], np.array(cors['a'])*100, label='a')
control = np.array(cors['o'])*100 + np.array(cors['s'])*100 + np.array(cors['a'])*100
ax.plot(cor_res['times'], control, label='control', color='gray')
ax.plot(cor_res['times'], osa_cors, label='o-s-a')
fig.tight_layout()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Variance exlained (%)')
ax.legend()
fig.tight_layout()

img_path = res_folder + f'sub_{sub_format}_control.png'
plt.savefig(img_path)
plt.clf()

############################ Plot  3 ######################################
fig, ax = plt.subplots(1,3, dpi=300, figsize=(30,10))

# Plot 1: oa shared
ax[0].set_title(f'oa shared')
ax[0].plot(cor_res['times'], oa_shared, label='o-a shared')
ax[0].plot(cor_res['times'], np.array(cors['s'])*100, label='s model')
ax[0].plot(cor_res['times'], u_o_cors, label='unique o')
ax[0].plot(cor_res['times'], u_a_cors, label='unique a')
ax[0].plot(cor_res['times'], np.array(cors['o'])*100, label='o model')
ax[0].plot(cor_res['times'], np.array(cors['a'])*100, label='a model')
ax[0].axvline(x=0, color='gray', alpha=0.5, linestyle='--')
ax[0].axvline(x=3, color='gray', alpha=0.5, linestyle='--')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Variance exlained (%)')
ax[0].legend()

# Plot 2: scenes
ax[1].set_title(f'os shared')
ax[1].plot(cor_res['times'], os_shared, label='o-s shared')
ax[1].plot(cor_res['times'], np.array(cors['a'])*100, label='a model')
ax[1].plot(cor_res['times'], u_o_cors, label='unique o')
ax[1].plot(cor_res['times'], u_s_cors, label='unique s')
ax[1].plot(cor_res['times'], np.array(cors['o'])*100, label='o model')
ax[1].plot(cor_res['times'], np.array(cors['s'])*100, label='s model')
ax[1].axvline(x=0, color='gray', alpha=0.5, linestyle='--')
ax[1].axvline(x=3, color='gray', alpha=0.5, linestyle='--')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Variance exlained (%)')
ax[1].legend()

# Plot 3: actions
ax[2].set_title(f'sa shared')
ax[2].plot(cor_res['times'], sa_shared, label='s-a shared')
ax[2].plot(cor_res['times'], np.array(cors['o'])*100, label='o model')
ax[2].plot(cor_res['times'], u_s_cors, label='unique s')
ax[2].plot(cor_res['times'], u_a_cors,label='unique a')
ax[2].plot(cor_res['times'], np.array(cors['s'])*100, label='s model')
ax[2].plot(cor_res['times'], np.array(cors['a'])*100, label='a model')
ax[2].axvline(x=0, color='gray', alpha=0.5, linestyle='--')
ax[2].axvline(x=3, color='gray', alpha=0.5, linestyle='--')
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Variance exlained (%)')
ax[2].legend()

fig.tight_layout()
img_path = res_folder + f'sub_{sub_format}_shared.png'
plt.savefig(img_path)
plt.clf()

