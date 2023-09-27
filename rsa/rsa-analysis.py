import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
from sklearn.metrics import pairwise_distances

models = ['ft', 'cb_ft', 'skip_ft', 'bert', 'guse']
model = 'guse'

with open(f'/scratch/azonneveld/meta-explore/{model}_wv.pkl', 'rb') as f: #maybe change filepath of wv?
    wv_dict = pickle.load(f)

res_folder = '/scratch/azonneveld/rsa/plots' 

zets = ['train', 'test']
vars = ['objects', 'scenes', 'actions']

maxs = []
mins = []
rdms = []

for j in range(len(zets)):
    zet = zets[j]

    for i in range(len(vars)):
        var = vars[i]

        print(f"Creating RDM {zet} {var}")
        c_dict = wv_dict[zet][var]
        keys = list(c_dict.keys())
        n_features = c_dict[keys[0]].shape[0]
        n_stimuli = len(c_dict.keys())

        f_matrix = np.zeros((n_stimuli, n_features))
        for k in range(n_stimuli):
            key = keys[k]
            f_matrix[k, :] = c_dict[key]

        metric = 'euclidean'
        rdm = pairwise_distances(f_matrix, metric=metric)
        rdms.append(rdm)
        maxs.append(np.max(rdm))
        mins.append(np.min(rdm))

        # save rdms (only triangle?)
len_rdms = len(rdms)
print(f"len rdms {len_rdms}")

fig, ax = plt.subplots(2,3, dpi = 300)
fig.suptitle(f'{model} RDMs')
rdm_count = 0

for j in range(len(zets)):
    zet = zets[j]

    for i in range(len(vars)):
        var = vars[i]
        rdm = rdms[rdm_count]

        print(f"rdm_count {rdm_count}")
        ax[j, i].set_title(f'{zet} {var}', fontsize=8)
        sns.heatmap(rdm, annot=False, yticklabels=False, xticklabels=False, ax=ax[j, i], cbar=True, cmap="viridis", square=True)
        ax[j, i].set_xlabel("", fontsize=10)
        ax[j, i].set_ylabel("", fontsize=10)

        rdm_count += 1

fig.tight_layout()
img_path = res_folder + f'/{model}_rdms.png'
plt.savefig(img_path)
plt.clf()