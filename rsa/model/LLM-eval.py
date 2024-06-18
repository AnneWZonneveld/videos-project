""""
Early on exploration of the semantic structure as established through different 
language models. RDMs not constructed as n conditions x n conditions but as 
n of unique labels x n of unique labels. 

Run after having extracted embeddings for the different models (using scripts in /meta-explore).

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
from sklearn.cluster import KMeans, AgglomerativeClustering
import tensorflow as tf
import tensorflow_hub as hub
from datasets import load_dataset, Dataset, DatasetDict
from natsort import index_natsorted
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

file_path = '/scratch/azonneveld/downloads/annotations_humanScenesObjects.json'
md = pd.read_json(file_path).transpose()

zets = ['train', 'test']
vars = ['objects', 'scenes', 'actions']
models = ['ft', 'cb_ft', 'skip_ft', 'bert', 'guse']
metric='euclidean'
res_folder = '/scratch/azonneveld/rsa/model/plots' 


# ---------------- RDM type 1: all labels (not per video)
def calc_t1_rdms(): 
    """ Calculates n of unique labels x n of unique labels RDMs for 
    different models. 
    """

    rdm_models = {}

    for model in models:

        print(f'Model {model}')
        
        # Load embeddings
        with open(f'/scratch/azonneveld/meta-explore/{model}_wv.pkl', 'rb') as f: #maybe change filepath of wv?
            wv_dict = pickle.load(f)
        
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

            rdm_zets[zet] = rdm_vars
        
        rdm_models[model] = rdm_zets

    # Save all model rdms
    with open(f'/scratch/azonneveld/rsa/model/rdms/rdm_t1_models.pkl', 'wb') as f:
                    pickle.dump(rdm_models, f)

def calc_t1_rdm_freq():
    """
    RDMs based on GUSE model embeddings sorted on frequency. 
    """
        
    # Load embeddings
    with open(f'/scratch/azonneveld/meta-explore/guse_wv.pkl', 'rb') as f: #maybe change filepath of wv?
        wv_dict = pickle.load(f)
    
    # Load frequency data
    with open('/scratch/azonneveld/meta-explore/freq_data.pkl', 'rb') as f:
        freq_dict = pickle.load(f)

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

            freq_data = freq_dict[zet][var]
            f_matrix = np.zeros((n_labels, n_features))

            for k in range(n_labels):
                key = freq_data['label'].iloc[k]
                f_matrix[k, :] = c_dict[key]

            rdm = pairwise_distances(f_matrix, metric=metric)
            rdm_vars[var] = rdm

        rdm_zets[zet] = rdm_vars
    
    # Save all model rdms
    with open(f'/scratch/azonneveld/rsa/model/rdms/rdm_t1_guse_freq_sorted.pkl', 'wb') as f:
                    pickle.dump(rdm_zets, f)
    

    # Get max and min of all rdms 
    max_min_dict ={}
    maxs = []
    mins = []
    for zet in zets:
        for var in vars:
            rdm = rdm_zets[zet][var]
            max = np.max(rdm)
            min = np.max(rdm)
            maxs.append(max)
            mins.append(min)
    max = np.max(np.asarray(maxs))
    min = np.min(np.asarray(mins))
    max_min_dict['max'] = max
    max_min_dict['min'] = min

    # Create plots
    fig, ax = plt.subplots(2,3, dpi = 500)
    fig.suptitle(f'Type 1 GUSE RDMs , freq sorted')

    for j in range(len(zets)):
        zet = zets[j]

        for i in range(len(vars)):
            var = vars[i]

            rdm = rdm_zets[zet][var]

            im = ax[j, i].imshow(rdm, vmin=0, vmax=max)
            ax[j, i].set_title(f'{zet} {var}', fontsize=8)
            ax[j, i].set_xlabel("", fontsize=10)
            ax[j, i].set_ylabel("", fontsize=10)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.ax.set_ylabel(f'{metric} distance', fontsize=12)

    img_path = res_folder + f'/rdm_t1_guse_freq_sorted.png'
    plt.savefig(img_path)
    plt.clf()
    

def plot_t1_rdms():
    "Plotting t1 model RDMs."

    # Load rdms
    if sorted == 'True':
        with open(f'/scratch/azonneveld/rsa/model/rdms/rdm_t1_models_sorted.pkl', 'rb') as f: 
            rdm_models = pickle.load(f)
    else:
        with open(f'/scratch/azonneveld/rsa/model/rdms/rdm_t1_models.pkl', 'rb') as f: 
            rdm_models = pickle.load(f)

    # Get max and min of all rdms per model
    model_ultimates = {}
    for model in models:
        max_min_dict ={}
        model_maxs = []
        model_mins = []
        for zet in zets:
            for var in vars:
                rdm = rdm_models[model][zet][var]
                max = np.max(rdm)
                min = np.max(rdm)
                model_maxs.append(max)
                model_mins.append(min)
        model_max = np.max(np.asarray(model_maxs))
        model_min = np.min(np.asarray(model_mins))
        max_min_dict['max'] = model_max
        max_min_dict['min'] = model_min
        model_ultimates[model] = max_min_dict

    # Create model plots
    for model in models:

        fig, ax = plt.subplots(2,3, dpi = 300)
        fig.suptitle(f'Type 1 {model} RDMs, sorted={sorted}')

        max_dist = model_ultimates[model]['max']
        min_dist = model_ultimates[model]['min']

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

        if sorted == True:
            img_path = res_folder + f'/{model}_rdms_sorted.png'
        else:
            img_path = res_folder + f'/{model}_rdms.png'
        plt.savefig(img_path)
        plt.clf()


def rdm_model_sim():
    """
    Calculate the similarity between the semantic structures as established through different
    semantic models.
    """

    # Load rdms
    with open(f'/scratch/azonneveld/rsa/model/rdms/rdm_models.pkl', 'rb') as f: 
        rdm_models = pickle.load(f)

    fig, ax = plt.subplots(2,3, dpi = 300)
    fig.suptitle(f'Pairwise correlation Type 1 model RDMs')

    for j in range(len(zets)):
        zet = zets[j]

        for i in range(len(vars)):
            var = vars[i]

            df = pd.DataFrame()
            for k in range(len(models)):
                model =  models[k]
                rdm = squareform(rdm_models[model][zet][var].round(5))
                df[model] = rdm

            im = ax[j,i].imshow(df.corr(), vmin=-0.5, vmax=1)
            ax[j,i].set_xticks([0,1,2,3,4]) 
            ax[j,i].set_xticklabels(models, fontsize=5)
            ax[j,i].set_yticks([0,1,2,3,4]) 
            ax[j,i].set_yticklabels(models, fontsize=5)
            ax[j,i].set_title(f'{zet} {var}', fontsize=7)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.ax.set_ylabel(f'Pearson correlation', fontsize=12)

    img_path = res_folder + f'/rdm_t1_cors.png'
    plt.savefig(img_path)
    plt.clf()

            
#  --------------  MAIN
# Type 1 RDM analysis
# calc_t1_rdms()
# plot_t1_rdms()
# rdm_model_sim()
# calc_t1_rdm_freq()



