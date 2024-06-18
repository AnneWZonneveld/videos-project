"""
Helper functions for creating model RDMs and extra analyses. 
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
import tqdm

file_path = '/scratch/azonneveld/downloads/annotations_humanScenesObjects.json'
md = pd.read_json(file_path).transpose()
vars = ['objects', 'scenes', 'actions']

def load_glob_md(ob_type = 'freq'):
    new_md = md.copy().reset_index(drop=True)

    for var in vars:

        if var != 'objects':
            with open(f'/scratch/azonneveld/rsa/model/global_embs/{var}', 'rb') as f: 
                global_df = pickle.load(f)
        else:
            with open(f'/scratch/azonneveld/rsa/model/global_embs/{var}_{ob_type}', 'rb') as f: 
                global_df = pickle.load(f)

        new_md = pd.concat([new_md, global_df], axis=1)
    
    return new_md


def load_glob_md_v2(emb_type='avg'):
    """ 
    Load glob md for model-rdms-v2.py

    """
    
    new_md = md.copy().reset_index(drop=True)

    for var in vars:
            
        with open(f'/scratch/azonneveld/rsa/model/global_embs/{var}_{emb_type}', 'rb') as f: 
            global_df = pickle.load(f)

        new_md = pd.concat([new_md, global_df], axis=1)
    
    return new_md


def corr_nullDist(rdm_1, rdm_2, iterations=100):
    """
        Calculates null distribution based on permutations. 
    """
    print("Constructing null distribution")

    rdm_corr_null = []
    for i in range(iterations):
        if i%10 == 0:
            print('Iteration: ' + str(i) )

        # Create a random index that respects the structure of an rdm.
        shuffle = np.random.choice(rdm_1.shape[0], rdm_1.shape[0],replace=False)

        # shuffle RDM consistently for both dims
        shuffled_rdm_1 = rdm_1[shuffle,:] # rows
        shuffled_rdm_1 = rdm_1[:,shuffle] # columns

        # correlating with neural similarty matrix
        rdm_corr_null.append(spearmanr(squareform(shuffled_rdm_1, checks=False), squareform(rdm_2, checks=False))[0])

    return rdm_corr_null