""" 
Initial explorative analysis about what type of 'summary' embeddings to use in the computation 
of video x video RDM, i.e. global, derived (based on derived labels based on average embedding) or first label of set.
Additionally, specifically in the case of object labels, as the data structure is different, exploration of 
whether to use the average embedding, embedding of the most frequent label or sentence embedding (based on all object labels).

These RDMs are actually not used in further analyses; therefore we used model-rmds-v2.py

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
from model_rdm_utils import load_glob_md, corr_nullDist

file_path = '/scratch/azonneveld/downloads/annotations_humanScenesObjects.json'
md = pd.read_json(file_path).transpose()

zets = ['train', 'test']
vars = ['objects', 'scenes', 'actions']
models = ['ft', 'cb_ft', 'skip_ft', 'bert', 'guse']
metric='euclidean'
res_folder = '/scratch/azonneveld/rsa/model/plots' 

def pearsonr_pval(x,y):
        return pearsonr(x,y)[1]


# # ------------------- RDM type 2: per video 
def global_emb(var, ob_type='freq'):
    """
    Calculate global embeddings for specific visual event component.
    
    var: str
        Visual event component 'objects'/'scenes'/'actions'
    ob_type: str
       Embedding type specifically used for objects embeddings; 'freq', 'avg', 'sentence'
    """

    print(f'Calcualting gloabal embeddings {var} {ob_type}')
         
    # Load embeddings
    with open(f'/scratch/azonneveld/meta-explore/guse_wv_all.pkl', 'rb') as f: 
        wv_dict_all = pickle.load(f)

    with open(f'/scratch/azonneveld/meta-explore/guse_wv.pkl', 'rb') as f: 
        wv_dict = pickle.load(f)

    n_features = 512

    # Load model
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)

    global_df = pd.DataFrame()

    if var in ['actions', 'scenes']:

        # Setting up var-specific FAISS dataset
        print (f'Setting up FAISS {var}')
        var_labels = list(wv_dict['train'][var].keys()) + list(wv_dict['test'][var].keys())   
        var_labels = list(np.unique(np.asarray(var_labels)))                  

        ds_dict = {'labels': var_labels}
        ds = Dataset.from_dict(ds_dict)
        embeddings_ds = ds.map(
            lambda x: {"embeddings": model([x['labels']]).numpy()[0]}
        )
        embeddings_ds.add_faiss_index(column='embeddings')

        # Calc global emb
        print(f'calculating global emb {var}')
        global_embs = []
        global_labels = []
        global_deriveds = []

        for i in range(len(md)):
            labels = md[var].iloc[i]

            label_matrix = np.zeros((len(labels), n_features))
            for i in range(len(labels)):
                label = labels[i]
                label_matrix[i, :] = wv_dict_all[label]
            global_emb = np.average(label_matrix, axis=0).astype(np.float32)

            # Get closest neighbour of global embedding = global label & get embedding of that global label = global derived
            score, sample = embeddings_ds.get_nearest_examples("embeddings", global_emb, k=1)
            global_label = sample['labels'][0]
            global_derived = wv_dict_all[global_label]

            global_embs.append(global_emb)
            global_labels.append(global_label)
            global_deriveds.append(global_derived)

        emb_col = 'glb_' + var + '_emb'
        lab_col = 'glb_' + var + '_lab'
        der_col = 'glb_' + var + '_der'
        global_df[emb_col] = global_embs
        global_df[lab_col] = global_labels
        global_df[der_col] = global_deriveds
    
        # Save new_df
        with open(f'/scratch/azonneveld/rsa/model/global_embs/{var}', 'wb') as f:
            pickle.dump(global_df, f)

    elif var == 'objects':

        if ob_type == 'freq':

            shared_max_count = 0
            shared_max_ids = []

            # Calc global emb
            global_embs = []
            global_labels = []

            for i in range(len(md)):
                labels = md[var].iloc[i]
                
                # Get most frequent object label
                labels_list = []
                for j in range(len(labels)):
                    single_obs = labels[j]

                    for obs in single_obs:
                        if obs != '--':
                            labels_list.append(obs)
                
                values, counts = np.unique(labels_list, return_counts=True)             
                global_label = values[np.argmax(counts)]
                global_labels.append(global_label)

                # Check for shared maxs
                maximum = np.max(counts)
                max_ids = np.where(counts==maximum)[0]
                if len(max_ids) > 1:
                    shared_max_count = shared_max_count + 1
                    shared_max_ids.append(i)

                global_emb = wv_dict_all[global_label] 
                global_embs.append(global_emb)
            
            emb_col = 'glb_' + var + '_emb'
            lab_col = 'glb_' + var + '_lab'
            global_df[emb_col] = global_embs
            global_df[lab_col] = global_labels
        
            print(f"shared max count {shared_max_count}")
        
        elif ob_type == 'sentence':
            
            # Setting up var-specific FAISS dataset
            print (f'Setting up FAISS {var}')
            var_labels = list(wv_dict['train'][var].keys()) + list(wv_dict['test'][var].keys())   
            var_labels = list(np.unique(np.asarray(var_labels)))  

            ds_dict = {'labels': var_labels}
            ds = Dataset.from_dict(ds_dict)
            embeddings_ds = ds.map(
                lambda x: {"embeddings": model([x['labels']]).numpy()[0]}
            )
            embeddings_ds.add_faiss_index(column='embeddings')  

            # Calc global emb
            global_embs = []
            global_labels = []
            global_deriveds = []

            # For every video in dataset
            for i in range(len(md)):
                labels = md[var].iloc[i]
                
                obs_embs = np.zeros((len(labels), 512))

                for j in range(len(labels)):
                    single_obs = labels[j]
                    labels_list = []

                    for obs in single_obs:
                        if obs != '--':
                            labels_list.append(obs)
                    
                    # Get 'sentence' embedding per observer
                    if len(labels_list) > 0:
                        sentence = [" ".join(labels_list)]
                    else:
                        sentence= labels_list
                    emb = model(sentence).numpy()[0]
                    obs_embs[j, :] = emb
                
                # Average over observers
                global_emb = np.mean(obs_embs, axis=0).astype(np.float32)

                # Get nearest neighbour for global embedding
                score, sample = embeddings_ds.get_nearest_examples("embeddings", global_emb, k=1)
                global_label = sample['labels'][0] 
                global_derived = wv_dict_all[global_label]

                global_embs.append(global_emb)
                global_labels.append(global_label)
                global_deriveds.append(global_derived)

            emb_col = 'glb_' + var + '_emb'
            lab_col = 'glb_' + var + '_lab'
            der_col = 'glb_' + var + '_der'
            global_df[emb_col] = global_embs
            global_df[lab_col] = global_labels
            global_df[der_col] = global_deriveds

        # Save new_df
        with open(f'/scratch/azonneveld/rsa/model/global_embs/objects_{ob_type}', 'wb') as f:
            pickle.dump(global_df, f)



def calc_t2_rdms(emb_type='global', sort=False,  ob_type = 'freq'): 
    """
    Compute n videos x n videos RDM based on specified embedding type.

    Emb_type: str
        Overal used embeding type; 'global', 'derived', 'first'
    Sort: bool
        Create sorted vs non-sorted RDMs
    Ob_type: str
        Embedding type specifically used for objects embeddings; 'freq', 'avg', 'sentence'

    """

    print(f"Calculating t2 rdms {emb_type} {ob_type}")
    
    # Load global embeddings df 
    md_global = load_glob_md(ob_type=ob_type)

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
        file_name = 'guse_glb'
    elif emb_type == 'derived':
        file_name = 'guse_der'
    elif emb_type == 'first':
        file_name = 'guse_first'
    
    if sort == True:
        with open(f'/scratch/azonneveld/rsa/model/rdms/rdm_t2_{file_name}_{ob_type}_sorted.pkl', 'wb') as f:
                    pickle.dump(rdm_zets, f)
    else:
        with open(f'/scratch/azonneveld/rsa/model/rdms/rdm_t2_{file_name}_{ob_type}.pkl', 'wb') as f:
            pickle.dump(rdm_zets, f)
    
    # Save feature matrices
    with open(f'/scratch/azonneveld/rsa/fm_{file_name}_{ob_type}.pkl', 'wb') as f:
        pickle.dump(fm_zets, f)
    

def plot_t2_rdms(emb_type='global', sort=False, ob_type='freq'):

    print(f"Plotting t2 rdms {emb_type} {ob_type}")

    # Load rdms
    if emb_type == 'global':
        file_name = 'guse_glb'
    elif emb_type == 'derived':
        file_name = 'guse_der'
    elif emb_type == 'first':
        file_name = 'guse_first'
 
    if sort == True:   
        with open(f'/scratch/azonneveld/rsa/model/rdms/rdm_t2_{file_name}_{ob_type}_sorted.pkl', 'rb') as f: 
            rdms = pickle.load(f)
    else:
        with open(f'/scratch/azonneveld/rsa/model/rdms/rdm_t2_{file_name}_{ob_type}.pkl', 'rb') as f: 
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
    fig.suptitle(f'Type 2 GUSE RDMs {emb_type}, sorted {sort}, ob_type={ob_type}', size=8)

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
        img_path = res_folder + f'/rdm_t2/rdm_t2_guse_{emb_type}_{ob_type}_sorted.png'
    else:
        img_path = res_folder + f'/rdm_t2/rdm_t2_guse_{emb_type}_{ob_type}.png'
    plt.savefig(img_path)
    plt.clf()


def rdm_t2_obj_sim():
    """
    Sanity check to see how much the semantic space of 
    'frequency object labels' vs the 'sentence' object labels differ.

    """

    with open(f'/scratch/azonneveld/rsa/model/rdms/rdm_t2_guse_glb_freq.pkl', 'rb') as f: 
        freq_rdms = pickle.load(f)
    
    with open(f'/scratch/azonneveld/rsa/model/rdms/rdm_t2_guse_glb_sentence.pkl', 'rb') as f: 
        sentence_rdms = pickle.load(f)
    
    with open(f'/scratch/azonneveld/rsa/model/fm_guse_glb_sentence.pkl', 'rb') as f: 
        sentence_fm = pickle.load(f)
    
    with open(f'/scratch/azonneveld/rsa/model/fm_guse_glb_freq.pkl', 'rb') as f: 
        freq_fm = pickle.load(f)

    # Sanity check --> see whether feature matrices differ   
    train_cor = pearsonr(freq_fm['train']['objects'].ravel(), sentence_fm['train']['objects'].ravel())
    test_cor  = pearsonr(freq_fm['test']['objects'].ravel(), sentence_fm['test']['objects'].ravel())
    print(f'fm cor:')
    print(f'train: {train_cor}')
    print(f'test: {test_cor}')
          
    # RDM cor
    train_cor = pearsonr(squareform(freq_rdms['train']['objects']), squareform(sentence_rdms['train']['objects']))
    test_cor  = pearsonr(squareform(freq_rdms['test']['objects']), squareform(sentence_rdms['test']['objects']))
    print(f'rdm cor:')
    print(f'train: {train_cor}')
    print(f'test: {test_cor}')


def rdm_t2_label_sim(ob_type='freq', its=1000):
    """
    Correlate object/scene/action space with videoxvideo RDM for GUSE model. 

    ob_type: str
       Embedding type specifically used for objects embeddings; 'freq', 'avg', 'sentence'.
    its: int
        Nr of iterations used to compute significance.

    """

    with open(f'/scratch/azonneveld/rsa/model/rdms/rdm_t2_guse_glb_{ob_type}.pkl', 'rb') as f: 
            rdms = pickle.load(f)
    
    fig, ax = plt.subplots(1, 2, dpi = 300)
    fig.suptitle(f'Pairwise correlation Type 2 RDMs, ob_type={ob_type}')

    for j in range(len(zets)):
        zet = zets[j]
        df = pd.DataFrame()

        for i in range(len(vars)):
            var = vars[i]
            rdm = squareform(rdms[zet][var].round(5))
            df[var] = rdm

        im = ax[j].imshow(df.corr(), vmin=0.0, vmax=1)
        ax[j].set_xticks([0,1,2]) 
        ax[j].set_xticklabels(vars, fontsize=5)
        ax[j].set_yticks([0,1,2]) 
        ax[j].set_yticklabels(vars, fontsize=5)
        ax[j].set_title(f'{zet} {var}', fontsize=7)

        ps = df.corr(method=pearsonr_pval).to_numpy().ravel()
        corrected_ps = multipletests(ps, alpha=0.05, method='bonferroni')[1] 
        corrected_ps = corrected_ps.reshape(3,3)
        print(f'{zet} label cor:')
        print(f'{df.corr()}')   
        
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.ax.set_ylabel(f'Pearson correlation', fontsize=12)

    img_path = res_folder + f'/rdm_t2/rdm_t2_label_cor_{ob_type}.png'
    plt.savefig(img_path)
    plt.clf()

    # Permutation tests
    fig, ax = plt.subplots(1, 3, dpi = 300)
    fig.suptitle(f'Permutation correlation test: its={its}')
    for i in range(len(vars)):
        var_1 = vars[i]

        if i == len(vars) - 1:
            var_2 = vars[0]
        else:
            var_2 = vars[i+1]
        
        print(f'{var_1} x {var_2}')

        rdm_1 = rdms['train'][var_1]
        rdm_2 = rdms['train'][var_2]
        rdm_cor = pearsonr(squareform(rdm_1, checks=False), squareform(rdm_2, checks=False))[0]
        rdm_corr_null = corr_nullDist(rdm_1, rdm_2, iterations=its)
        p_val = np.mean(rdm_corr_null>rdm_cor) 
        print(f'cor: {rdm_cor}')
        print(f'p val: {p_val}')

        fig, ax = plt.subplots()
        sns.histplot(np.array(rdm_corr_null))
        ax.set_xlabel("Cor")
        ax.set_title(f"{var_1} x {var_2}, its={its}")
        ax.axvline(rdm_cor, color="red")
    
        img_path = res_folder + f'/rdm_t2/permutations/{var_1}_x_{var_2}.png'
        plt.savefig(img_path)
        plt.clf()


def rdm_t2_emb_type_sim():
    """
    Correlate semantic space for different type of 'summary labels' 
    to see how they differ, i.e. 'global', 'derived', 'first' labels.
    """

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

        with open(f'/scratch/azonneveld/rsa/model/rdms/{file_name}', 'rb') as f: 
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




    
#  --------------  MAIN
# Type 2 RDM analysis
# global_emb('objects', ob_type='freq')
# global_emb('objects', ob_type='sentence')
# calc_t2_rdms('global', sort=True, ob_type='freq')
# calc_t2_rdms('global', sort=True, ob_type='sentence')
# calc_t2_rdms('global', sort=False, ob_type='freq')
# calc_t2_rdms('global', sort=False, ob_type='sentence')
# calc_t2_rdms('derived', sort=True)
# calc_t2_rdms('global', sort=False)
# calc_t2_rdms('derived', sort=False)
# calc_t2_rdms('first')
# plot_t2_rdms('global', sort=False, ob_type='freq')
# plot_t2_rdms('global', sort=False, ob_type='sentence')
# rdm_t2_obj_sim()
# rdm_t2_label_sim(ob_type='freq', its=10000)
# rdm_t2_label_sim(ob_type='sentence')
# plot_t2_rdms('derived', sort=True)
# plot_t2_rdms('global', sort=False)
# plot_t2_rdms('derived', sort=False)
# plot_t2_rdms('first')
# rdm_t2_emb_type_sim()