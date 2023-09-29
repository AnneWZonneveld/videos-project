import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
from sklearn.metrics import pairwise_distances

file_path = '/scratch/giffordale95/projects/eeg_videos/videos_metadata/annotations.json'
md = pd.read_json(file_path).transpose()

res_folder = '/scratch/azonneveld/rsa/plots' 

zets = ['train', 'test']
vars = ['objects', 'scenes', 'actions']


# ---------------- RDM type 1: all labels (not per video)
models = ['ft', 'cb_ft', 'skip_ft', 'bert', 'guse']
# model = 'guse'
metric = 'euclidean'

rdm_models = {}

for model in models:

    with open(f'/scratch/azonneveld/meta-explore/{model}_wv.pkl', 'rb') as f: #maybe change filepath of wv?
        wv_dict = pickle.load(f)

    rdms = []

    # Calculate rdms
    rdm_zets = {}
    for j in range(len(zets)):
        zet = zets[j]
        rdm_vars = {}

        for i in range(len(vars)):
            var = vars[i]
            
            print(f"Creating RDM {zet} {var}")
            c_dict = wv_dict[zet][var]
            keys = list(c_dict.keys())
            n_features = c_dict[keys[0]].shape[0]
            n_labels = len(c_dict.keys())

            f_matrix = np.zeros((n_labels, n_features))
            for k in range(n_labels):
                key = keys[k]
                f_matrix[k, :] = c_dict[key]

            rdm = pairwise_distances(f_matrix, metric=metric)
            rdm_vars[var] = rdm
            rdms.append(rdm)
            maxs.append(np.max(rdm))
            mins.append(np.min(rdm))

        rdm_zets[zet] = rdm_vars
    
    rdm_models[model] = rdm_zets

with open(f'/scratch/azonneveld/rsa/rdm_models.pkl', 'wb') as f:
            pickle.dump(rdm_models, f)

# Create plot
max_dist = np.max(np.asarray(maxs))
min_dist = np.min(np.asarray(mins))

for model in models:

    fig, ax = plt.subplots(2,3, dpi = 300)
    fig.suptitle(f'{model} RDMs')

    for j in range(len(zets)):
        zet = zets[j]

        for i in range(len(vars)):
            var = vars[i]

            rdm = rdm_models[model][zet][var]

            im = ax[j, i].imshow(rdm, vmin=0, vmax=max_dist)
            ax[j, i].set_title(f'{zet} {var}', fontsize=8)
            ax[j, i].set_xlabel("", fontsize=10)
            ax[j, i].set_ylabel("", fontsize=10)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.ax.set_ylabel(f'{metric} distance', fontsize=12)

    img_path = res_folder + f'/{model}_rdms.png'
    plt.savefig(img_path)
    plt.clf()




# # ------------------- RDM type 2: per video 
# for zet in zets:
    
#     for var in vars:

#         c_md = md[md['set']==zet][var]
#         c_dict = wv_dict[zet][var]

#         keys = list(c_dict.keys())
#         n_features = c_dict[keys[0]].shape[0]
#         n_stimuli = len(c_md)
        
#         f_matrix = np.zeros((n_stimuli, n_features))
