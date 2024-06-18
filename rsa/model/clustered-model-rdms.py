"""
Explorative analysis focusing on creating higher order RDMs based on
- the centroids of clustering analyses. 
- THINGs higher order labels.
First should run clustering analyses. 

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


class AgglomerativeClusteringWithPredict(AgglomerativeClustering):
    def __init__(self, fname, lname):
        super().__init__(fname, lname) 

    def __init__(self, n_clusters, distance_threshold, linkage, compute_full_tree, compute_distances):
        super().__init__(n_clusters=n_clusters, distance_threshold=distance_threshold, 
                         linkage=linkage, compute_full_tree=compute_full_tree, compute_distances=compute_distances)
        
    def predict(self, fm):
        self.labels_ = super().fit_predict(fm)

    def calc_centroids(self, fm):
        clf = NearestCentroid()
        clf.fit(fm, self.labels_)
        centroids = clf.centroids_
        self.cluster_centers_ = centroids
    
    def fit_predict(self, fm):
        self.labels_ = super().fit_predict(fm)
        self.calc_centroids(fm)

        return self

    cluster_centers_ = []
    labels_ = []



# ------------- RDM type 3 (cluster based)

def rdm_t3(ctype='hierarch', k=60, sorted=False, ob_type='freq', plot=False):
    """
        Build higher order RDM based on centroids of specified clustering fit. 
    """
    
    # Load global embeddings df 
    md_global = load_glob_md(ob_type=ob_type)

    # Select only train set
    zet ='train'
    md_select = md_global[md_global['set']==zet]
    n_stimuli = len(md_select)
    
    centroid_matrices = {}
    rdms = {}
    orderings = {}
    centroids_dict = {}
    
    if ctype=='hierarch':
        linkage = '_ward'
    else:
        linkage=''

    for var in vars:

        # Load cluster data
        if var == 'objects':
            with open(f'/scratch/azonneveld/clustering/fits/{ctype}_sample_train_{var}{linkage}_{ob_type}.pkl', 'rb') as f: 
                all_clusters = pickle.load(f)
        else:
            with open(f'/scratch/azonneveld/clustering/fits/{ctype}_sample_train_{var}{linkage}.pkl', 'rb') as f: 
                all_clusters = pickle.load(f)
        
        clusters = all_clusters[k]

        cluster_labels = clusters.labels_   
        centroids = []
        for i in range(n_stimuli):
            clus_label = cluster_labels[i]
            centroid = clusters.cluster_centers_[clus_label, :]
            centroids.append(centroid)
        
        if ctype=='kmean':
            centroids_dict[var] = centroids

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
            order = np.asarray(md_select.index)

        elif sorted =='biggest':
            clusters_unique, cluster_counts = np.unique(np.array(cluster_labels), return_counts=True)
            count_df = pd.DataFrame()
            count_df['cluster'] = clusters_unique
            count_df['size'] = cluster_counts
            count_df = count_df.sort_values(by=['size'], ascending=False)
            size_index = [*range(k)]
            count_df['size_index'] = size_index

            size_ids = []
            for i in range(n_stimuli):
                clus = md_select[col_1].iloc[i]
                size_id = count_df[count_df['cluster'] == clus]['size_index'].to_numpy()[0]
                size_ids.append(size_id)
            
            col_3 = var + '_size_id'
            col_4 = 'glb_' + var + '_lab'
            md_select[col_3] = size_ids
            md_select = md_select.sort_values(by=[col_3, col_4],
                    )
            
            order = np.asarray(md_select.index)

        for i in range(n_stimuli):
            centroid = md_select[col_2].iloc[i]
            centroid_matrix[i, :] = centroid
        
        rdm = pairwise_distances(centroid_matrix, metric='euclidean')
        rdms[var] = rdm
        centroid_matrices[var] = centroid_matrix

        if sorted != False:
            orderings[var] = order

    if sorted == True:
        with open(f'/scratch/azonneveld/rsa/model/rdms/t3/rdm_t3_{ctype}_k{k}_sorted', 'wb') as f:
            pickle.dump(rdms, f)
        with open(f'/scratch/azonneveld/clustering/ordering/{ctype}_k{k}_sorted', 'wb') as f:
            pickle.dump(orderings, f)
    elif sorted == 'biggest':
        with open(f'/scratch/azonneveld/rsa/model/rdms/t3/rdm_t3_{ctype}_k{k}_biggest', 'wb') as f:
            pickle.dump(rdms, f)
        with open(f'/scratch/azonneveld/clustering/ordering/{ctype}_k{k}_biggest', 'wb') as f:
            pickle.dump(orderings, f)
    else:
        with open(f'/scratch/azonneveld/rsa/model/rdms/t3/rdm_t3_{ctype}_k{k}', 'wb') as f:
            pickle.dump(rdms, f)
    
    if ctype == 'kmean':
        with open(f'/scratch/azonneveld/clustering/centroids/{ctype}_k{k}.pkl', 'wb') as f:
            pickle.dump(centroids_dict, f)


    if plot == True:

        # Get max and min of all rdms 
        maxs = []
        mins = []
        for var in vars:
            rdm = rdms[var]
            max = np.max(rdm)
            min = np.max(rdm)
            maxs.append(max)
            mins.append(min)
        max = np.max(np.asarray(maxs))
        min = np.min(np.asarray(mins))

        # Create plots
        fig, ax = plt.subplots(1, 2, dpi = 500)
        fig.suptitle(f'Type 3 HO RDMs {ctype} k={k}, sorted={sorted}')

        for i in range(len(vars)):
            var = vars[i]
            rdm = rdms[var]
    
            im = ax[i].imshow(rdm, vmin=0, vmax=1.4) #vmax = 1.4 to compare with basic-order
            ax[i].set_title(f'train {var}', fontsize=8)
            ax[i].set_xlabel("", fontsize=10)
            ax[i].set_ylabel("", fontsize=10)

        fig.tight_layout()
        cbar = fig.colorbar(im, ax=ax.ravel().tolist())
        cbar.ax.set_ylabel(f'Euclidean distance', fontsize=12)

        if sorted == True:
            img_path = res_folder + f'/rdm_t3/rdm_t3_{ctype}_k{k}_sorted.png'
        elif sorted == 'biggest':
            img_path = res_folder + f'/rdm_t3/rdm_t3_{ctype}_k{k}_biggest.png'
        else:
            img_path = res_folder + f'/rdm_t3/rdm_t3_{ctype}_k{k}.png'
        plt.savefig(img_path)
        plt.clf()

def THINGS_rdm(ob_type='freq', sorted=False):
    """
    Create higher order RDM based on labels of 27 categories of THINGS database. 
    """

    file_path = '/scratch/azonneveld/downloads/things_concepts.tsv'
    things_df = pd.read_table(file_path)

    md_global = load_glob_md(ob_type=ob_type)
    zet = 'train'
    md_select = md_global[md_global['set']==zet]
    n_stimuli = len(md_select)

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)

    categories = np.unique(things_df['All Bottom-up Categories'].dropna())

    # Check if category embeddings already present
    try:
        with open(f'/scratch/azonneveld/rsa/model/global_embs/THINGS', 'rb') as f: 
            cat_embeddings = pickle.load(f)    
    except:
    
        # Get embeddings for all categories; if multiple labels, only take first
        cat_embeddings = {}
        for cat in categories:
            cat = cat.split(',')[0]
            embedding = model([cat]).numpy().squeeze()
            cat_embeddings[cat] = embedding

        # Save category embeddings
        with open(f'/scratch/azonneveld/rsa/model/global_embs/THINGS', 'wb') as f:
            pickle.dump(cat_embeddings, f)
    

    # For each video, find higher order embedding
    ho_embeddings = []
    ho_final_labels = []
    counter = 0
    for i in range(n_stimuli):
        global_label = md_select.iloc[i]['glb_objects_lab']

        try:
            ho_labels = things_df[things_df['Word'] == global_label]['All Bottom-up Categories'].tolist()[0].split(',')
            counter = counter + 1
        except:
            pass

        ho_label = ho_labels[0]
        ho_embedding = cat_embeddings[ho_label]
        ho_embeddings.append(ho_embedding)
        ho_final_labels.append(ho_label)
    
    print(f'multiple ho_labels: {counter}')

    md_select['ho_embeddings'] = ho_embeddings
    md_select['ho_labels'] = ho_final_labels

    # Sort RDM on derived global label
    if sorted == True:
        sort_var = 'ho_labels'
        md_select = md_select.sort_values(by=sort_var,
                            key=lambda x: np.argsort(index_natsorted(md_select[sort_var]))
                            )

    n_features = ho_embedding.shape[0]          
    f_matrix = np.zeros((n_stimuli, n_features))

    for i in range(n_stimuli):
        embedding = md_select['ho_embeddings'].iloc[i]
        f_matrix[i, :] = embedding
    
    rdm = pairwise_distances(f_matrix, metric='euclidean')

    if sorted == True:
        with open(f'/scratch/azonneveld/rsa/model/rdms/t3/rdm_t3_things_sorted', 'wb') as f:
            pickle.dump(rdm, f)
    else:
        with open(f'/scratch/azonneveld/rsa/model/rdms/t3/rdm_t3_things', 'wb') as f:
            pickle.dump(rdm, f)

    # Create plots
    fig, ax = plt.subplots(1, 1, dpi = 500)
    ax.set_title(f'HO THINGS RDM, sorted={sorted}')

    im = ax.imshow(rdm)
    ax.set_xlabel("", fontsize=10)
    ax.set_ylabel("", fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(f'Euclidean distance', fontsize=12)
    fig.tight_layout()

    if sorted == True:
        img_path = res_folder + f'/rdm_t3/rdm_t3_things_sorted.png'
    else:
        img_path = res_folder + f'/rdm_t3/rdm_t3_things.png'

    plt.savefig(img_path)
    plt.clf()


  
def rdm_t3_sim(k=60):

    """
    Calculates the similarity between HO rdms for different clustering methods over different ks. 
    """

    # Calculate similarity between cluster-size based sorted kmeans and hierarchical rdms
    ks = [*range(10, 210, 10)]  
    cor_dict = {}
    n_features = 512
    n_stimuli  = 1000

    with open('/scratch/azonneveld/rsa/model/rdms/rdm_t2_guse_glb_freq.pkl', 'rb') as f:
        og_rdms = pickle.load(f)
    og_rdms = og_rdms['train']

    cor_dict = {}
    for var in vars:

        og_rdm = og_rdms[var]
        cors_var_dict = {}
        cors_var_dict['og-k'] = []
        cors_var_dict['og-h'] = []
        cors_var_dict['k-h'] = []

        for k in ks:

            with open(f'/scratch/azonneveld/rsa/model/rdms/t3/rdm_t3_hierarch_k{k}', 'rb') as f:
                h_rdms = pickle.load(f)
            
            with open(f'/scratch/azonneveld/rsa/model/rdms/t3/rdm_t3_kmean_k{k}', 'rb') as f:
                k_rdms = pickle.load(f)
        
            h_rdm = h_rdms[var]
            k_rdm = k_rdms[var]

            cors_var_dict['og-k'].append(pearsonr(squareform(og_rdm, checks=False), squareform(k_rdm, checks=False))[0])
            cors_var_dict['og-h'].append(pearsonr(squareform(og_rdm, checks=False), squareform(h_rdm, checks=False))[0])
            cors_var_dict['k-h'].append(pearsonr(squareform(k_rdm, checks=False), squareform(h_rdm, checks=False))[0])

        cor_dict[var] = cors_var_dict


    fig, ax = plt.subplots(1, 3,  dpi = 300, sharey=True, figsize=(15,5))
    for i in range(len(vars)):
        var = vars[i]

        var_dict = cor_dict[var]
        ax[i].plot(ks, var_dict['og-k'], label='og-k', color='red', marker='.')
        ax[i].plot(ks, var_dict['og-h'], label='og-h', color='green', marker='.')
        ax[i].plot(ks, var_dict['k-h'], label='k-h', color='blue', marker='.')
        ax[i].set_title(f'{var}', fontsize=7)
        ax[i].set_xlabel("K", fontsize=10)

        if var == 'actions':
            line_val = 179
            ax[i].axvline(x=line_val, color='gray', label=f'n of unique labels ({line_val})')
        elif var == 'scenes':
            line_val = 189
            ax[i].axvline(x=line_val, color='gray', label=f'n of unique labels ({line_val})')
                    
        if i == 0:
            ax[i].set_ylabel("Pearson cor", fontsize=10)

    fig.suptitle(f'Clustering methods similarities', size=9)
    fig.legend(labels=['og-k', 'og-h', 'k-h'])
    fig.tight_layout()
    
    img_path = res_folder + f'/rdm_t3/rdm_t3_corplot.png'
    plt.savefig(img_path)
    plt.clf()


def rdm_t3_THINGS_sim():

    """
    Calculates the similarity between HO rdms for different clustering methods compared to THINGS
    """

    # Calculate similarity between cluster-size based sorted kmeans and hierarchical rdms
    ks = [*range(10, 210, 10)]  
    n_features = 512
    n_stimuli  = 1000

    with open('/scratch/azonneveld/rsa/model/rdms/rdm_t2_guse_glb_freq.pkl', 'rb') as f:
        og_rdms = pickle.load(f)
    og_rdm = og_rdms['train']['objects']

    with open('/scratch/azonneveld/rsa/model/rdms/t3/rdm_t3_things', 'rb') as f:
        things_rdm = pickle.load(f)

    cor_dict = {}
    cor_dict['things-k'] = []
    cor_dict['things-h'] = []

    for k in ks:

        with open(f'/scratch/azonneveld/rsa/model/rdms/t3/rdm_t3_hierarch_k{k}', 'rb') as f:
            h_rdms = pickle.load(f)
        
        with open(f'/scratch/azonneveld/rsa/model/rdms/t3/rdm_t3_kmean_k{k}', 'rb') as f:
            k_rdms = pickle.load(f)
    
        h_rdm = h_rdms['objects']
        k_rdm = k_rdms['objects']

        cor_dict['things-k'].append(pearsonr(squareform(k_rdm, checks=False), squareform(things_rdm, checks=False))[0])
        cor_dict['things-h'].append(pearsonr(squareform(h_rdm, checks=False), squareform(things_rdm, checks=False))[0])

    cor_dict['things-og'] = pearsonr(squareform(og_rdm, checks=False), squareform(things_rdm, checks=False))[0]

    fig, ax = plt.subplots(1, 1,  dpi = 300)
    ax.axhline(cor_dict['things-og'], label='things-og', color='red' )
    ax.plot(ks, cor_dict['things-k'], label='things-k', color='green', marker='.')
    ax.plot(ks, cor_dict['things-h'], label='things-h', color='blue', marker='.')
    ax.set_title(f'THINGS vs clustering methods similarity', fontsize=7)
    ax.set_xlabel("K", fontsize=10)
    ax.set_ylabel("Pearson cor", fontsize=10)
    fig.legend(labels=['things-og', 'things-k', 'things-h'])
    fig.tight_layout()
    
    img_path = res_folder + f'/rdm_t3/rdm_t3_things_corplot.png'
    plt.savefig(img_path)
    plt.clf()

    
#  --------------  MAIN

# Type 3 RDM analysis
# ks = [*range(10, 210, 10)]  
# for k in ks:
#     rdm_t3(ctype='kmean', k=k)
#     rdm_t3(ctype='hierarch', k=k)

# rdm_t3_sim()

# THINGS_rdm()
# rdm_t3_THINGS_sim()


