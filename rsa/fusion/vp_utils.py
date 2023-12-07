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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import time
from multiprocessing import shared_memory


def corr_nullDist(rdm_cor, rdm_1, rdm_2, its=100):
    """
        Calculates null distribution based on permutations. 
    """
    print("Constructing null distribution")

    rdm_corr_null = []
    for i in tqdm(range(its)):

        # Create a random index that respects the structure of an rdm.
        shuffle = np.random.choice(rdm_1.shape[0], rdm_1.shape[0],replace=False)

        # shuffle RDM consistently for both dims
        shuffled_rdm_1 = rdm_1[shuffle,:] # rows
        shuffled_rdm_1 = rdm_1[:,shuffle] # columns

        # correlating with neural similarty matrix
        rdm_corr_null.append(spearmanr(squareform(shuffled_rdm_1, checks=False), squareform(rdm_2, checks=False))[0])
    
    # Calc p value
    p_val = np.mean(rdm_corr_null>rdm_cor) 

    return p_val


def calc_rsquared(t, data_shape, feature_rdm, shm, its=10):

    print(f'Calculating r^2 {t}')

    existing_shm = shared_memory.SharedMemory(name=shm)
    GA_rdms = np.ndarray(data_shape, dtype='float32', buffer=existing_shm.buf)

    tic = time.time()

    rdm_cors = []
    rdm_cor_ps = []

    # Calc correlation
    neural_rdm = GA_rdms[:, :, t]
    rdm_cor = spearmanr(squareform(neural_rdm, checks=False), squareform(feature_rdm, checks=False))[0] 
    rdm_cors.append(rdm_cor)

    # Calc significance
    rdm_p = corr_nullDist(rdm_cor, neural_rdm, feature_rdm, its=its) # testing purpose
    rdm_cor_ps.append(rdm_p)

    toc = time.time()

    print(f'iteration {t} in {toc-tic}')

    return np.array(rdm_cors, rdm_cor_ps)


def cor_variability(t, GA_rdms, feature_rdm, its=100):

    neural_rdm = GA_rdms[:, :, t]

    rdm_corr_boots = []

    for i in range(its):

        # Create a random index that respects the structure of an rdm.
        sample = np.random.choice(neural_rdm.shape[0], neural_rdm.shape[0],replace=True) 

        # Subsample from both the reference and the feature RDM
        neural_rdm_sample = neural_rdm[sample] 
        feature_rdm_sample = feature_rdm[sample] 

        # correlating with neural similarty matrix
        rdm_corr_boots.append(spearmanr(squareform(neural_rdm_sample, checks=False), squareform(feature_rdm_sample, checks=False))[0])

    # Get 95% confidenc interval
    lower_p = np.percentile(rdm_corr_boots, 5)
    upper_p = np.percentile(rdm_corr_boots, 95)

    return np.array(lower_p, upper_p)
