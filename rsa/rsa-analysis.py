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
# import tensorflow_hub as hub
from datasets import load_dataset, Dataset, DatasetDict
from natsort import index_natsorted

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
        global_deriveds = []

        for i in range(len(md)):
            labels = md[var].iloc[i]

            label_matrix = np.zeros((len(labels), n_features))
            for i in range(len(labels)):
                label = labels[i]
                label_matrix[i, :] = wv_dict[label]
            global_emb = np.average(label_matrix, axis=0).astype(np.float32)

            # Get closest neighbour of global embedding = global label + get embedding of global label = global derived
            score, sample = embeddings_ds.get_nearest_examples("embeddings", global_emb, k=1)
            global_label = sample['labels'][0]
            global_derived = wv_dict[global_label]

            global_embs.append(global_emb)
            global_labels.append(global_label)
            global_deriveds.append(global_derived)

        emb_col = 'glb_' + var + '_emb'
        lab_col = 'glb_' + var + '_lab'
        der_col = 'glb_' + var + '_der'
        new_md[emb_col] = global_embs
        new_md[lab_col] = global_labels
        new_md[der_col] = global_deriveds
    
    # Save new_df
    with open(f'/scratch/azonneveld/rsa/md_global.pkl', 'wb') as f:
                pickle.dump(new_md, f)
        

def calc_t2_rdms(emb_type='global', sort=False): 
    """
    Emb_type: 'global', 'derived', 'first'

    """
    print(f"Calculating t2 rdms {emb_type}")
    
    # Load global embeddings df
    with open(f'/scratch/azonneveld/rsa/md_global.pkl', 'rb') as f: 
        md_global = pickle.load(f)

    # Load embeddings
    with open(f'/scratch/azonneveld/meta-explore/guse_wv_all.pkl', 'rb') as f: 
        wv_dict = pickle.load(f)
    
    # Calculate RDMS per set per variable
    rdm_zets = {}
    fm_zets = {}
    n_features = 512

    for zet in zets:

        rdm_vars = {}
        fm_vars = {}
        zet_md = md_global[md_global['set']==zet]
        n_stimuli = len(zet_md)

        for var in vars:

            # Sort RDM on derived global label
            if sort == True:
                sort_var = 'glb_' + var +'_lab'
                zet_md = zet_md.sort_values(by=sort_var,
                                    key=lambda x: np.argsort(index_natsorted(zet_md[sort_var]))
                                    )

            f_matrix = np.zeros((n_stimuli, n_features))

            if emb_type == 'global':
                var_emb = 'glb_' + var +'_emb'
            elif emb_type == 'derived':
                var_emb = 'glb_' + var +'_der'
            
            if emb_type != 'first':
                for k in range(n_stimuli):
                    f_matrix[k, :] = zet_md[var_emb].iloc[k]
            else:
                for k in range(n_stimuli):
                    label = zet_md[var].iloc[k][0]
                    f_matrix[k, :] = wv_dict[label]

            rdm = pairwise_distances(f_matrix, metric=metric)
            rdm_vars[var] = rdm
            fm_vars[var] = f_matrix

        rdm_zets[zet] = rdm_vars
        fm_zets[zet] = fm_vars
    
    # Save rdms
    if emb_type == 'global':
        file_name = 'guse_glb.pkl'
    elif emb_type == 'derived':
        file_name = 'guse_der.pkl'
    elif emb_type == 'first':
        file_name = 'guse_first.pkl'
    
    if sort == True:
        with open(f'/scratch/azonneveld/rsa/rdm_t2_{file_name}_sorted', 'wb') as f:
                    pickle.dump(rdm_zets, f)
    else:
        with open(f'/scratch/azonneveld/rsa/rdm_t2_{file_name}', 'wb') as f:
            pickle.dump(rdm_zets, f)
    
    # Save feature matrices
    with open(f'/scratch/azonneveld/rsa/feature_m_{file_name}', 'wb') as f:
        pickle.dump(fm_zets, f)
    

def plot_t2_rdms(emb_type='global', sort=False):
    print(f"Plotting t2 rdms {emb_type}")

    # Load rdms
    if emb_type == 'global':
        file_name = 'rdm_t2_guse_glb.pkl'
    elif emb_type == 'derived':
        file_name = 'rdm_t2_guse_der.pkl'
    elif emb_type == 'first':
        file_name = 'rdm_t2_guse_first.pkl'

    if sort == True:   
        with open(f'/scratch/azonneveld/rsa/{file_name}_sorted', 'rb') as f: 
            rdms = pickle.load(f)
    else:
        with open(f'/scratch/azonneveld/rsa/{file_name}', 'rb') as f: 
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
    fig.suptitle(f'Type 2 GUSE RDMs {emb_type}, sorted {sort} ')

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

    if sort == True:  
        img_path = res_folder + f'/rdm_t2_guse_{emb_type}_sorted.png'
    else:
        img_path = res_folder + f'/rdm_t2_guse_{emb_type}.png'
    plt.savefig(img_path)
    plt.clf()


def rdm_t2_emb_type_sim():
    emb_types = ['global', 'derived', 'first']

    # Load rdms
    rdms_embtypes = {}
    for emb_type in emb_types:
        if emb_type == 'global':
            file_name = 'rdm_t2_guse_glb.pkl'
        elif emb_type == 'derived':
            file_name = 'rdm_t2_guse_der.pkl'
        elif emb_type == 'first':
            file_name = 'rdm_t2_guse_first.pkl'

        with open(f'/scratch/azonneveld/rsa/{file_name}', 'rb') as f: 
            rdms = pickle.load(f)
        
        rdms_embtypes[emb_type] = rdms

    fig, ax = plt.subplots(2,3, dpi = 300)
    fig.suptitle(f'Pairwise correlation Type 2 emb types')

    for j in range(len(zets)):
        zet = zets[j]

        for i in range(len(vars)):
            var = vars[i]

            df = pd.DataFrame()
            for k in range(len(emb_types)):
                emb_type =  emb_types[k]
                rdm = squareform(rdms_embtypes[emb_type][zet][var].round(5))
                df[emb_type] = rdm

            im = ax[j,i].imshow(df.corr(), vmin=0.2, vmax=1)
            # im = sns.heatmap(df.corr(), vmin=-0.5, vmax=1, ax=ax[j,i], square=True, cbar=False, cmap='viridis', annot=True)
            ax[j,i].set_xticks([0,1,2]) 
            ax[j,i].set_xticklabels(emb_types, fontsize=5)
            ax[j,i].set_yticks([0,1,2]) 
            ax[j,i].set_yticklabels(emb_types, fontsize=5)
            ax[j,i].set_title(f'{zet} {var}', fontsize=7)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.ax.set_ylabel(f'Pearson correlation', fontsize=12)

    img_path = res_folder + f'/rdm_t2_embtypes_cor.png'
    plt.savefig(img_path)
    plt.clf()


# ------------- RDM type 3 (cluster based)
# Load clusters

def rdm_type3(k=40, sorted=False):
    
    # Load global embeddings df
    with open(f'/scratch/azonneveld/rsa/md_global.pkl', 'rb') as f: 
        md_global = pickle.load(f)

    # Select only train set
    zet ='train'
    md_select = md_global[md_global['set']==zet]
    n_stimuli = len(md_select)
    vars_oi = ['scenes', 'actions']
    
    centroid_matrices = {}
    rdms = {}

    for var in vars_oi:

        # Load cluster data
        with open(f'/scratch/azonneveld/clustering/kmean_k{k}_train_{var}.pkl', 'rb') as f: 
            clusters = pickle.load(f)

        cluster_labels = clusters.labels_   
        centroids = []
        for i in range(n_stimuli):
            clus_label = cluster_labels[i]
            centroid = clusters.cluster_centers_[clus_label, :]
            centroids.append(centroid)
        
        col_1 = var + '_clust'
        col_2 = var + '_centroid'
        md_select[col_1] = cluster_labels
        md_select[col_2] = centroids

        n_features = centroid.shape[0]          
        centroid_matrix = np.zeros((n_stimuli, n_features))

        if sorted == True:
            sort_col1 = col_1
            sort_col2 = 'glb_' + var + '_lab'
            md_select = md_select.sort_values(by=[sort_col1, sort_col2],
                    )
            
        for i in range(n_stimuli):
            centroid = md_select[col_2].iloc[i]
            centroid_matrix[i, :] = centroid
        
        rdm = pairwise_distances(centroid_matrix, metric='euclidean')
        rdms[var] = rdm
        centroid_matrices[var] = centroid_matrix

    if sorted == True:
        with open(f'/scratch/azonneveld/rsa/rdm_t3_sorted', 'wb') as f:
                    pickle.dump(rdms, f)
    else:
        with open(f'/scratch/azonneveld/rsa/rdm_t3', 'wb') as f:
            pickle.dump(rdms, f)

        
    # Get max and min of all rdms 
    maxs = []
    mins = []
    for var in vars_oi:
        rdm = rdms[var]
        max = np.max(rdm)
        min = np.max(rdm)
        maxs.append(max)
        mins.append(min)
    max = np.max(np.asarray(maxs))
    min = np.min(np.asarray(mins))

    # Create plots
    fig, ax = plt.subplots(1, 2, dpi = 500)
    fig.suptitle(f'Type 3 higher-order RDMs, sorted={sorted}')

    for i in range(len(vars_oi)):
        var = vars_oi[i]
        rdm = rdms[var]
 
        im = ax[i].imshow(rdm, vmin=0, vmax=1.4) #vmax = 1.4 to compare with basic-order
        ax[i].set_title(f'train {var}', fontsize=8)
        ax[i].set_xlabel("", fontsize=10)
        ax[i].set_ylabel("", fontsize=10)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.ax.set_ylabel(f'Euclidean distance', fontsize=12)

    if sorted == True:
        img_path = res_folder + f'/rdm_t3_sorted.png'
    else:
        img_path = res_folder + f'/rdm_t3.png'
    plt.savefig(img_path)
    plt.clf()



#  --------------  MAIN
# Type 1 RDM analysis
# calc_t1_rdms()
# plot_t1_rdms()
# rdm_model_sim()

# Type 2 RDM analysis
# global_emb()
# calc_t2_rdms('global', sort=True)
# calc_t2_rdms('derived', sort=True)
calc_t2_rdms('global', sort=False)
# calc_t2_rdms('derived', sort=False)
# calc_t2_rdms('first')
# plot_t2_rdms('global', sort=True)
# plot_t2_rdms('derived', sort=True)
# plot_t2_rdms('global', sort=False)
# plot_t2_rdms('derived', sort=False)
# plot_t2_rdms('first')
# rdm_t2_emb_type_sim()



# test
# md_cols_oi = ['objects', 'glb_objects_lab'] 
# md_cols_oi = ['scenes', 'glb_scenes_lab'] 
# md_cols_oi = ['actions', 'glb_actions_lab']
# md_select = md_global[md_cols_oi].iloc[120:130]



