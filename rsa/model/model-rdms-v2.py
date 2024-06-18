"""

Final code used to compute the n video x n video RDMs that are used for furhter analyses.
Additional exploration of average vs most frequent label based RDMs.

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
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from model_rdm_utils import load_glob_md_v2, corr_nullDist
import random 
from matplotlib.colors import LogNorm 

file_path = '/scratch/azonneveld/downloads/annotations_humanScenesObjects.json'
md = pd.read_json(file_path).transpose()

zets = ['train', 'test']
vars = ['objects', 'scenes', 'actions']
metric='euclidean'
res_folder = f'/scratch/azonneveld/rsa/model/plots/rdm_t2/' 


# # ------------------- RDM type 2: per video 
def global_emb(emb_type='avg'):
    """
    Compute the global embeddings for all videos 

    Emb_type: str
        'avg' or 'freq'
    
    """

    print(f'Calcualting global {emb_type} embedding')
         
    # Load embeddings
    with open(f'/scratch/azonneveld/meta-explore/guse_wv_all.pkl', 'rb') as f: 
        wv_dict_all = pickle.load(f)

    with open(f'/scratch/azonneveld/meta-explore/guse_wv.pkl', 'rb') as f: 
        wv_dict = pickle.load(f)

    n_features = 512

    # Load model
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)

    for var in vars:

        global_df = pd.DataFrame()

        if emb_type =='avg':
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


        global_embs = []
        global_labels = []
        global_deriveds = []

        shared_max_count = 0

        for i in range(len(md)):

            labels = md[var].iloc[i]

            if var == 'objects':

                labels_list = []
                for j in range(len(labels)):
                    single_obs = labels[j]

                    for obs in single_obs:
                        if obs != '--':
                            labels_list.append(obs)
                
                labels = labels_list

            if emb_type == 'avg': 
                
                # Calculate average embedding    
                label_matrix = np.zeros((len(labels), n_features))
                for i in range(len(labels)):
                    label = labels[i]
                    label_matrix[i, :] = wv_dict_all[label]
                global_emb = np.average(label_matrix, axis=0).astype(np.float32)

                # Get nearest neighbour
                score, sample = embeddings_ds.get_nearest_examples("embeddings", global_emb, k=1)
                global_label = sample['labels'][0]

                global_embs.append(global_emb)
                global_labels.append(global_label)

            elif emb_type == 'freq':

                values, counts = np.unique(labels, return_counts=True)             
                global_label = values[np.argmax(counts)]
                global_labels.append(global_label)

                # Check for shared maxs
                maximum = np.max(counts)
                max_ids = np.where(counts==maximum)[0]
                if len(max_ids) > 1:
                    shared_max_count = shared_max_count + 1

                global_emb = wv_dict_all[global_label] 
                global_embs.append(global_emb)
            
        print(f'shared max: {shared_max_count}')
        emb_col = 'glb_' + var + '_emb'
        lab_col = 'glb_' + var + '_lab'
        global_df[emb_col] = global_embs
        global_df[lab_col] = global_labels

        # Save new_df
        with open(f'/scratch/azonneveld/rsa/model/global_embs/{var}_{emb_type}', 'wb') as f:
            pickle.dump(global_df, f)
            


def calc_t2_rdms(emb_type='avg', sort=False, metric='euclidean'): 
    """
    Compute n videos x n videos RDMs for different visual event components.
    
    Emb_type: str
        'avg' or 'freq'
    Sort: bool
        Calculate sorted vs non-sorted RDM
    Metric: str
        'euclidean' or (1- pearson)'correlation'

    """

    print(f"Calculating t2 rdms {emb_type}")
    
    # Load global embeddings df 
    md_global = load_glob_md_v2(emb_type=emb_type)

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

            var_emb = 'glb_' + var +'_emb'

            f_matrix = np.zeros((n_stimuli, n_features))
            for k in range(n_stimuli):
                f_matrix[k, :] = zet_md[var_emb].iloc[k]

            rdm = pairwise_distances(f_matrix, metric=metric) ** 2 #should be squared? bc cv euclidean rdm is also squared
            rdm_vars[var] = rdm

        rdm_zets[zet] = rdm_vars

    # Save rdms
    if sort == True:
        sort_label = '_sorted'
    else:
        sort_label = ''
    
    folder = f'/scratch/azonneveld/rsa/model/rdms/t2/{metric}/'
    if os.path.isdir(folder) == False:
        os.makedirs(folder)

    file = folder + f'rdm_t2_{emb_type}{sort_label}.pkl'
    with open(file, 'wb') as f:
        pickle.dump(rdm_zets, f)

    
def plot_t2_rdms(emb_type='avg', sort=False):
    """
    Plot n videos x n videos RDMs

    Emb_type: str
        'avg' or 'freq'
    Sort: bool
        Calculate sorted vs non-sorted RDM

    """   

    print(f"Plotting t2 rdms {emb_type}")
    sns.set_style('white')
    sns.set_style("ticks")
    sns.set_context('paper', 
                    rc={'font.size': 14, 
                        'xtick.labelsize': 10, 
                        'ytick.labelsize':10, 
                        'axes.titlesize' : 13,
                        'figure.titleweight': 'bold', 
                        'axes.labelsize': 13, 
                        'legend.fontsize': 8, 
                        'font.family': 'Arial',
                        'axes.spines.right' : False,
                        'axes.spines.top' : False})
    sns.set_palette('viridis')

    # Load rdms
    if sort == True:
        sort_label = '_sorted'
    else:
        sort_label = ''
 
    with open(f'/scratch/azonneveld/rsa/model/rdms/t2/{metric}/rdm_t2_{emb_type}{sort_label}.pkl', 'rb') as f: 
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
    fig, ax = plt.subplots(2,3, dpi = 700)
    fig.suptitle(f'Semantic model RDMs')

    for j in range(len(zets)):
        zet = zets[j]

        for i in range(len(vars)):
            var = vars[i]

            rdm = rdms[zet][var]

            im = ax[j, i].imshow(rdm, vmin=0, vmax=max)
            ax[j, i].set_title(f'{var}')
            ax[j, i].set_xlabel("", fontsize=10)
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            if i in [0, 4]:
                if j == 0:
                    ax[j, i].set_ylabel("Train")
                else:
                    ax[j, i].set_ylabel("Test")
            else:
                ax[j, i].set_ylabel("", fontsize=10)
            sns.despine()

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8)
    cbar.ax.set_ylabel(f'{metric} distance')

    img_path = res_folder + f'rdm_t2_{emb_type}{sort_label}.png'
    plt.savefig(img_path)
    plt.clf()


def plot_t2_rdms_rank(emb_type='avg', sort=False):
    """
    Plot rank-normalized n videos x n videos RDMs.

    Emb_type: str
        'avg' or 'freq'
    Sort: bool
        Calculate sorted vs non-sorted RDM

    """   

    print(f"Plotting t2 rdms {emb_type}")
    sns.set_style('white')
    sns.set_style("ticks")
    sns.set_context('paper', 
                    rc={'font.size': 14, 
                        'xtick.labelsize': 10, 
                        'ytick.labelsize':10, 
                        'axes.titlesize' : 13,
                        'figure.titleweight': 'bold', 
                        'axes.labelsize': 13, 
                        'legend.fontsize': 8, 
                        'font.family': 'Arial',
                        'axes.spines.right' : False,
                        'axes.spines.top' : False})
    sns.set_palette('viridis')

    # Load rdms
    if sort == True:
        sort_label = '_sorted'
    else:
        sort_label = ''
 
    with open(f'/scratch/azonneveld/rsa/model/rdms/t2/{metric}/rdm_t2_{emb_type}{sort_label}.pkl', 'rb') as f: 
        rdms = pickle.load(f)

    # Create plots
    fig, ax = plt.subplots(2,3, dpi = 700)
    fig.suptitle(f'Semantic model RDMs')

    for j in range(len(zets)):
        zet = zets[j]

        for i in range(len(vars)):
            var = vars[i]

            rdm = rdms[zet][var]
            
            max_value = np.max(rdm)
            normalized_rdm = (rdm / max_value) * 100

            im = ax[j, i].imshow(normalized_rdm, vmin=0, vmax=100, cmap='viridis')
            ax[j, i].set_title(f'{var}')
            ax[j, i].set_xlabel("", fontsize=10)
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            if i in [0, 4]:
                if j == 0:
                    ax[j, i].set_ylabel("Train")
                else:
                    ax[j, i].set_ylabel("Test")
            else:
                ax[j, i].set_ylabel("", fontsize=10)

            ax[j, i].spines['top'].set_visible(False)
            ax[j, i].spines['bottom'].set_visible(False)
            ax[j, i].spines['right'].set_visible(False)
            ax[j, i].spines['left'].set_visible(False)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.9)
    cbar.ax.set_ylabel('Normalized Dissimilarity (%)')

    img_path = res_folder + f'rdm_t2_{emb_type}_norm{sort_label}.png'
    plt.savefig(img_path)
    plt.clf()



def rdm_t2_emb_type_sim():
    """
        Check if rdms for different emb_types are similar.
        For avg embedding based RDM, most frequent embedding based RDM 
        and permuted frequency based RDM.
    """

    # Load data 
    with open(f'/scratch/azonneveld/rsa/model/rdms/t2/rdm_t2_freq.pkl', 'rb') as f: 
        freq_rdms = pickle.load(f)
    
    with open(f'/scratch/azonneveld/rsa/model/rdms/t2/rdm_t2_avg.pkl', 'rb') as f: 
        avg_rdms = pickle.load(f)
    
    permuted_rdms = {}
    var_dict = {}
    for var in vars:

        with open(f'/scratch/azonneveld/rsa/model/rdms/t2/rdm_t2_GA_{var}.pkl', 'rb') as f: 
            permuted_rdm = pickle.load(f)
        
        permuted_rdm = permuted_rdm['GA_rdm']
        
        var_dict[var] = permuted_rdm

    permuted_rdms['train'] = var_dict


    # Compare different methods
    methods = ['avg', 'freq', 'perm_freq']
    method_rdms = {
        'avg': avg_rdms,
        'freq': freq_rdms,
        'perm_freq': permuted_rdms
    }

    # For each combination of methods
    for i in range(len(methods)):
        method_1 = methods[i]

        if i == len(methods) - 1:
            method_2 = methods[0]
        else:
            method_2 = methods[i+1]
        
        print(f'----- {method_1} x {method_2} -------')

        rdms_m1 = method_rdms[method_1]
        rdms_m2 = method_rdms[method_2]

        # Correlate RDMs for different vars
        for var in vars:

            rdm_m1 = rdms_m1['train'][var]
            rdm_m2 = rdms_m2['train'][var]

            cor = spearmanr(squareform(rdm_m1, checks=False), squareform(rdm_m2, checks=False))

            print(f'{var}: {cor}')



def rdm_t2_space_sim(emb_type='avg', its=10000, test=False, data_split='test'):
    """
    Correlate object/scene/action space with videoxvideo RDM for GUSE model. 

    emb_type: str
        'avg' / 'freq' / 'perm_freq'
    its: int
        N of iterations used for calcualtion of significance
    test: bool
        Whether to calculate significance y/n
    data_split: str
        'train' / 'test'
    """

    if emb_type =='perm_freq':
        
        permuted_rdms = {}
        var_dict = {}
        for var in vars:

            with open(f'/scratch/azonneveld/rsa/model/rdms/t2/{metric}/rdm_t2_GA_{var}.pkl', 'rb') as f: 
                permuted_rdm = pickle.load(f)
            
            permuted_rdm = permuted_rdm['GA_rdm']
            
            var_dict[var] = permuted_rdm

        permuted_rdms[data_split] = var_dict
        rdms = permuted_rdms
       
        fig, ax = plt.subplots(dpi=300)
        fig.suptitle(f'Pairwise correlation Type 2 RDMs, emb_typ={emb_type}')

        zet = data_split
        df = pd.DataFrame()

        for i in range(len(vars)):
            var = vars[i]
            rdm = squareform(rdms[zet][var].round(5))
            df[var] = rdm

        im = ax.imshow(df.corr(method='spearman'), vmax=1)
        ax.set_xticks([0,1,2]) 
        ax.set_xticklabels(vars, fontsize=10)
        ax.set_yticks([0,1,2]) 
        ax.set_yticklabels(vars, fontsize=10)
        ax.set_title(f'{zet}', fontsize=7)

        fig.tight_layout()
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(f'Spearman correlation', fontsize=12)

        img_path = res_folder + f'rdm_t2_label_cor_{emb_type}_{data_split}.png'
        plt.savefig(img_path)
        plt.clf()
    
    else: 

        with open(f'/scratch/azonneveld/rsa/model/rdms/t2/{metric}/rdm_t2_{emb_type}.pkl', 'rb') as f: 
                rdms = pickle.load(f)

        # Calc vmin
        vmin_values = []  
        for j in range(len(zets)):
            zet = zets[j]
            df = pd.DataFrame()

            for i in range(len(vars)):
                var = vars[i]
                rdm = squareform(rdms[zet][var].round(5))
                df[var] = rdm

            # Calculate the minimum correlation value
            vmin_values.append(df.corr(method='spearman').values.min())

        # Use the minimum value as vmin for all subplots
        vmin = min(vmin_values)

        fig, ax = plt.subplots(1, 2, dpi = 300)
        fig.suptitle(f'Pairwise correlations semantic RDMs')

        for j in range(len(zets)):
            zet = zets[j]
            df = pd.DataFrame()

            for i in range(len(vars)):
                var = vars[i]
                rdm = squareform(rdms[zet][var].round(5))
                df[var] = rdm
            
            corr_matrix = df.corr(method='spearman').values
            corr_matrix_tril = np.tril(corr_matrix)  # Get lower triangle

            im = ax[j].imshow(corr_matrix_tril, norm=LogNorm(vmin=vmin, vmax=1), cmap='viridis')

            # Annotate the heatmap with actual correlation values
            for i in range(len(vars)):
                for k in range(len(vars)):
                    text = f"{corr_matrix[i, k]:.3f}"
                    ax[j].text(k, i, text, ha='center', va='center', color='white' if corr_matrix[i, k] < 0.5 else 'black', fontsize=8)

            ax[j].set_xticks([0,1,2]) 
            ax[j].set_xticklabels(vars)
            ax[j].set_yticks([0,1,2]) 
            ax[j].set_yticklabels(vars)
            ax[j].set_title(f'{zet}')

            ax[j].spines['top'].set_visible(False)
            ax[j].spines['bottom'].set_visible(False)
            ax[j].spines['right'].set_visible(False)
            ax[j].spines['left'].set_visible(False)

        
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.5)
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.6)
        cbar.ax.set_ylabel(f'Log-norm Spearman correlation')

        img_path = res_folder + f'rdm_t2_label_cor_{emb_type}.png'
        plt.savefig(img_path)
        plt.clf()


    if test == True:
        # Permutation tests
        for zet in zets:
            for i in range(len(vars)):
                fig, ax = plt.subplots(dpi = 300)
                fig.suptitle(f'Permutation test: its={its}, emb_type={emb_type}')

                var_1 = vars[i]

                if i == len(vars) - 1:
                    var_2 = vars[0]
                else:
                    var_2 = vars[i+1]
                
                print(f'{var_1} x {var_2}')

                rdm_1 = rdms[zet][var_1]
                rdm_2 = rdms[zet][var_2]
                rdm_cor = spearmanr(squareform(rdm_1, checks=False), squareform(rdm_2, checks=False))[0]
                rdm_corr_null = corr_nullDist(rdm_1, rdm_2, iterations=its)
                p_val = np.mean(rdm_corr_null>rdm_cor) 
                print(f'cor: {rdm_cor}')
                print(f'p val: {p_val}')
                print(f'p val corrected: {p_val/3}')

                fig, ax = plt.subplots()
                sns.histplot(np.array(rdm_corr_null))
                ax.set_xlabel("Cor")
                ax.set_title(f"{var_1} x {var_2}, its={its}")
                ax.axvline(rdm_cor, color="red")
            
                img_path = res_folder + f'var_permutations/{var_1}_x_{var_2}_{emb_type}_{zet}.png'
                plt.savefig(img_path)
                plt.clf()
    else:

        for zet in zets:
            print(f'{zet}')
            for i in range(len(vars)):
                var_1 = vars[i]

                if i == len(vars) - 1:
                    var_2 = vars[0]
                else:
                    var_2 = vars[i+1]
                
                print(f'{var_1} x {var_2}')

                rdm_1 = rdms[zet][var_1]
                rdm_2 = rdms[zet][var_2]
                rdm_cor = spearmanr(squareform(rdm_1, checks=False), squareform(rdm_2, checks=False))[0]
                print(f'cor: {rdm_cor}')


def label_distances():
    """
    Evaluation of frequency based global labels. 
    - Checks how many times there is a share top-X among most frequent label per video
    - Checks across how many labels this 1st place is shared
    - Checks the average distance between the shared top-X labels
    """
         
    # Load embeddings
    with open(f'/scratch/azonneveld/meta-explore/guse_wv_all.pkl', 'rb') as f: 
        wv_dict_all = pickle.load(f)

    with open(f'/scratch/azonneveld/meta-explore/guse_wv.pkl', 'rb') as f: 
        wv_dict = pickle.load(f)

    for var in vars:

        info_df = pd.DataFrame()
        ids = []
        top_xs = []
        avg_distances = []

        shared_max_count = 0

        for i in range(len(md)):

            labels = md[var].iloc[i]

            if var == 'objects':

                labels_list = []
                for j in range(len(labels)):
                    single_obs = labels[j]

                    for obs in single_obs:
                        if obs != '--':
                            labels_list.append(obs)
                
                labels = labels_list

            values, counts = np.unique(labels, return_counts=True)             

            # Check for shared maxs
            maximum = np.max(counts)
            max_ids = np.where(counts==maximum)[0]

            if len(max_ids) > 1:
                shared_max_count = shared_max_count + 1

                top_x = max_ids.shape[0]
                top_x_labels = values[max_ids]

                top_x_embeddings = np.zeros((top_x, 512))
                for j in range(top_x):
                    label = top_x_labels[j]
                    embedding = wv_dict_all[label] 
                    top_x_embeddings[j] = embedding
                
                distances = squareform(pairwise_distances(top_x_embeddings, metric=metric), checks=False)
                avg_distance = np.mean(distances)

                ids.append(i)
                top_xs.append(top_x)
                avg_distances.append(avg_distance)
        
        info_df['ids'] = ids
        info_df['top_xs'] = top_xs
        info_df['avg_distances'] = avg_distances

        # Plot top x size & avg distances
        fig, ax = plt.subplots(1, 2,  dpi = 300, sharey=True)
        sns.histplot(data=info_df, x='top_xs', ax=ax[0], bins=4)
        ax[0].set_title(f'Top-x size (n={shared_max_count})')
        ax[0].set_xlabel('Size')
        sns.histplot(data=info_df, x='avg_distances', ax=ax[1])
        ax[1].set_title('Avg distance top-x')
        ax[1].set_xlabel(f'{metric} distance')
        fig.suptitle(f'{var}')
        fig.tight_layout()
    
        img_path = res_folder + f'freq_info_{var}'
        plt.savefig(img_path)
        plt.clf()


def permutation_analysis(var='objects', its=100):
    """
    Analysis that asses how much noise introduced by arbitrarly picking a 
    label from the top-x for a video. 
    - permutes the choice of label in situations of a top-x
    - build rdm
    - average rdm over all permutation
    - compare ga rdm to leave one out ga rdm 

    var: str
        'objects' 'scenes' 'actions'
    its: int
        how many iterations used to build ga rdm
    """

    print(f'Performing permutation analysis for {var}, its={its}')

    # Load embeddings
    with open(f'/scratch/azonneveld/meta-explore/guse_wv_all.pkl', 'rb') as f: 
        wv_dict = pickle.load(f)
    
    # Retrieve top-x for every video  
    zet_md = md[md['set']=='train']
    top_x_md = zet_md.copy().reset_index(drop=True)
    
    top_xs = [] 
    for i in range(len(top_x_md)):

        labels = top_x_md[var].iloc[i]

        if var == 'objects':

            labels_list = []
            for j in range(len(labels)):
                single_obs = labels[j]

                for obs in single_obs:
                    if obs != '--':
                        labels_list.append(obs)
            
            labels = labels_list

        values, counts = np.unique(labels, return_counts=True)    

        # Check for shared maxs
        maximum = np.max(counts)
        max_ids = np.where(counts==maximum)[0]
        top_x_labels = values[max_ids]
        top_xs.append(top_x_labels)
    
    col_name = var + '_top_x'
    top_x_md[col_name] = top_xs


    # Build rdms
    print('Building rdms')
    n_stimuli = len(top_x_md)
    rdms = np.zeros((n_stimuli, n_stimuli, its))

    for it in range(its):

        seed = it
        np.random.seed(seed)

        f_matrix = np.zeros((n_stimuli, 512))

        for k in range(n_stimuli):
            top_x_labels = top_x_md[col_name].iloc[k]
            pick_label = random.choice(top_x_labels)
            embedding  = wv_dict[pick_label]
            f_matrix[k, :] = embedding
        
        rdm = pairwise_distances(f_matrix, metric=metric)
        rdms[:, :, it] = rdm
    

    # Asses correlations between rdm permutations
    print('Assesing correlations between permutation rdms')
    counter = 0
    perm_cors = []
    for rdm_1_id in range(its):
        rdm_1 = rdms[:, :, rdm_1_id]
        for rdm_2_id in range(rdm_1_id):

            rdm_2 = rdms[:, :, rdm_2_id]
            cor = spearmanr(squareform(rdm_1, checks=False), squareform(rdm_2, checks=False))[0]
            perm_cors.append(cor)

            if counter % 100 == 0:
                print(f'counter: {counter}')

            counter = counter + 1

    perm_cors_df = pd.DataFrame()
    perm_cors_df['cors'] = perm_cors

    # Plot distribution of pairwise correlations between permuted rdms
    fig, ax = plt.subplots()
    sns.histplot(data=perm_cors_df, x='cors')
    ax.set_title(f'Cor permuted freq rdms {var}, its={its}')
    ax.set_xlabel('Spearman cor')

    img_path = res_folder + f'freq_permutations/cor_distr_{var}.png'
    plt.savefig(img_path)
    plt.clf()

    # Asses noise ceiling
    noiseLower = np.zeros(its)
    noiseHigher = np.zeros(its)
    GA_rdm = np.mean(rdms, axis=2)

    for it in range(its):
        it_rdm = rdms[:, :, it]
        noiseHigher[it]= spearmanr(squareform(it_rdm, checks=False), squareform(GA_rdm, checks=False))[0]
        
        mask = np.ones(its,dtype=bool)
        mask[it] = 0
        rdms_without = rdms[:, :, mask]

        GA_without = np.mean(rdms_without, axis=2)
        noiseLower[it] = spearmanr(squareform(it_rdm, checks=False), squareform(GA_without, checks=False))[0]

    noiseCeiling = {}
    noiseCeiling['UpperBound'] = np.mean(noiseHigher, axis=0)
    noiseCeiling['LowerBound'] = np.mean(noiseLower, axis=0)
    print(noiseCeiling)

    # Save as dictionary with extra info (its etc) + noise ceiling
    perm_dict = {
        'GA_rdm': GA_rdm,
        'its': its, 
        'noise_ceiling': noiseCeiling,
        'feature': var
    }

    with open(f'/scratch/azonneveld/rsa/model/rdms/t2/rdm_t2_GA_{var}.pkl', 'wb') as f:
        pickle.dump(perm_dict, f)


    
#  --------------  MAIN
# Type 2 RDM analysis
# global_emb(emb_type='avg')
# global_emb(emb_type='freq')

# calc_t2_rdms('avg', sort=False, metric=metric)
# calc_t2_rdms('freq', sort=False)

# plot_t2_rdms('avg', sort=False)
# plot_t2_rdms('freq', sort=False)

rdm_t2_space_sim(emb_type='avg', test=True)
# rdm_t2_space_sim(emb_type='freq')

# label_distances()
# permutation_analysis(var='objects', its=100) # test
# permutation_analysis(var='actions', its=100)
# permutation_analysis(var='scenes', its=100)

# rdm_t2_emb_type_sim()
# rdm_t2_space_sim(emb_type='perm_freq')
