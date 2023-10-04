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
from datasets import load_dataset, Dataset, DatasetDict

file_path = '/scratch/azonneveld/downloads/annotations_humanScenes.json'
md = pd.read_json(file_path).transpose()

res_folder = '/scratch/azonneveld/rsa/plots' 

zets = ['train', 'test']
vars = ['objects', 'scenes', 'actions']
models = ['ft', 'cb_ft', 'skip_ft', 'bert', 'guse']
metric='euclidean'

# ---------------- RDM type 1: all labels (not per video)
def calc_t1_rdms(): 

    rdm_models = {}

    for model in models:
        
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
    with open(f'/scratch/azonneveld/rsa/rdm_t1_models.pkl', 'wb') as f:
                pickle.dump(rdm_models, f)

def plot_t1_rdms():

    # Load rdms
    with open(f'/scratch/azonneveld/meta-explore/rdm_t1_models.pkl', 'rb') as f: 
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
        fig.suptitle(f'Type 1 {model} RDMs')

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

        img_path = res_folder + f'/{model}_rdms.png'
        plt.savefig(img_path)
        plt.clf()


def rdm_model_sim():

    # Load rdms
    with open(f'/scratch/azonneveld/rsa/rdm_models.pkl', 'rb') as f: 
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
            # im = sns.heatmap(df.corr(), vmin=-0.5, vmax=1, ax=ax[j,i], square=True, cbar=False, cmap='viridis', annot=True)
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

            
            
# # ------------------- RDM type 2: per video 
def global_emb():
         
    # Load embeddings
    with open(f'/scratch/azonneveld/meta-explore/guse_wv_all.pkl', 'rb') as f: # guse wv all is empty?
        wv_dict = pickle.load(f)

    keys = list(wv_dict.keys())
    n_features = wv_dict[keys[0]].shape[0]

    # Load model
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)

    # Set up FAISS dataset
    ds_dict = {'labels': wv_dict.keys()}
    ds = Dataset.from_dict(ds_dict)
    embeddings_ds = ds.map(
        lambda x: {"embeddings": model([x['labels']]).numpy()[0]}
    )
    embeddings_ds.add_faiss_index(column='embeddings')

    new_md = md.copy()

    for var in vars:
        print(f'calculating global emb {var}')
        global_embs = []
        global_labels = []

        for i in range(len(md)):
            labels = md[var].iloc[i]

            label_matrix = np.zeros((len(labels), n_features))
            for i in range(len(labels)):
                label = labels[i]
                label_matrix[i, :] = wv_dict[label]
            global_emb = np.average(label_matrix, axis=0).astype(np.float32)

            # Get closest neighbour of global embedding (vs get predicted based label?)
            score, sample = embeddings_ds.get_nearest_examples("embeddings", global_emb, k=1)
            global_label = sample['labels']

            global_embs.append(global_emb)
            global_labels.append(global_label)

        emb_col = 'glb_' + var + '_emb'
        lab_col = 'glb_' + var + '_lab'
        new_md[emb_col] = global_embs
        new_md[lab_col] = global_labels
    
    # Save new_df
    with open(f'/scratch/azonneveld/rsa/md_global.pkl', 'wb') as f:
                pickle.dump(new_md, f)
        

def calc_t2_rdms(): 
    print("Calculating t2 rdms")
    
    # Load global embeddings df
    with open(f'/scratch/azonneveld/rsa/md_global.pkl', 'rb') as f: 
        md_global = pickle.load(f)
    
    # Calculate RDMS per set per variable
    rdm_zets = {}
    for zet in zets:

        rdm_vars = {}
        for var in vars:

            var_gl_emb = 'glb_' + var +'_emb'
            var_gl_lab = 'glb_' + var +'_lab'

            c_md = md_global[md_global['set']==zet][var_gl_emb]

            n_features = c_md.iloc[0].shape[0]
            n_stimuli = len(c_md)
            
            f_matrix = np.zeros((n_stimuli, n_features))
            for k in range(n_stimuli):
                f_matrix[k, :] = c_md.iloc[k]

            rdm = pairwise_distances(f_matrix, metric=metric)
            rdm_vars[var] = rdm

        rdm_zets[zet] = rdm_vars
    
    # Save rdms
    with open(f'/scratch/azonneveld/rsa/rdm_t2_guse.pkl', 'wb') as f:
                pickle.dump(rdm_zets, f)

def plot_t2_rdms():
    print("Plotting t2 rdms")

    # Load rdms
    with open(f'/scratch/azonneveld/rsa/rdm_t2_guse.pkl', 'rb') as f: 
        rdms = pickle.load(f)

    # Get max and min of all rdms 
    max_min_dict ={}
    maxs = []
    mins = []
    for zet in zets:
        for var in vars:
            rdm = rdms[zet][var]
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
    fig.suptitle(f'Type 2 GUSE RDMs')

    for j in range(len(zets)):
        zet = zets[j]

        for i in range(len(vars)):
            var = vars[i]

            rdm = rdms[zet][var]

            im = ax[j, i].imshow(rdm, vmin=0, vmax=max)
            ax[j, i].set_title(f'{zet} {var}', fontsize=8)
            ax[j, i].set_xlabel("", fontsize=10)
            ax[j, i].set_ylabel("", fontsize=10)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.ax.set_ylabel(f'{metric} distance', fontsize=12)

    img_path = res_folder + f'/global_guse_rdms.png'
    plt.savefig(img_path)
    plt.clf()


def rdm_t2_sim():
    print("Calculating t2 rdm similarity")

    # Load rdms
    with open(f'/scratch/azonneveld/rsa/rdm_t2_guse.pkl', 'rb') as f: 
        rdms = pickle.load(f)

    fig, ax = plt.subplots(1, 2, dpi = 300)
    fig.suptitle(f'Pairwise correlation Type 2 GUSE RDMs')

    for j in range(len(zets)):
        zet = zets[j]

        df = pd.DataFrame()
        for i in range(len(vars)):
            var = vars[i]
            rdm = squareform(rdms[zet][var].round(5))
            df[var] = rdm

        im = ax[j].imshow(df.corr(), vmin=0, vmax=0.3)
        # im = sns.heatmap(df.corr(), vmin=-0.5, vmax=1, ax=ax[j,i], square=True, cbar=False, cmap='viridis', annot=True)
        ax[j].set_xticks([0,1,2]) 
        ax[j].set_xticklabels(vars, fontsize=5)
        ax[j].set_yticks([0,1,2]) 
        ax[j].set_yticklabels(vars, fontsize=5)
        ax[j].set_title(f'{zet}', fontsize=7)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.ax.set_ylabel(f'Pearson correlation', fontsize=12)

    img_path = res_folder + f'/rdm_guse_t2_cors.png'
    plt.savefig(img_path)
    plt.clf()


#  --------------  MAIN
# Type 1 RDM analysis
# calc_t1_rdms()
# plot_t1_rdms()
# rdm_model_sim()

# Type 2 RDM analysi
# global_emb()
# calc_t2_rdms()
# plot_t2_rdms()
rdm_t2_sim()

