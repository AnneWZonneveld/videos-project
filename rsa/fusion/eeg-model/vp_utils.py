import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
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
    for i in range(its):

        if i % 1000 == 0:
            print(f'null distr iteration: {i}')

        # Create a random index that respects the structure of an rdm.
        shuffle = np.random.choice(rdm_1.shape[0], rdm_1.shape[0],replace=False)

        # shuffle RDM consistently for both dims
        shuffled_rdm_1 = rdm_1[shuffle,:] # rows
        shuffled_rdm_1 = rdm_1[:,shuffle] # columns

        # correlating with neural similarty matrix
        rdm_corr_null.append(spearmanr(squareform(shuffled_rdm_1, checks=False), squareform(rdm_2, checks=False))[0])
    
    # Calc p value
    rdm_corr_null = np.array(rdm_corr_null)
    p_val = np.mean(rdm_corr_null>rdm_cor) 

    return p_val


def calc_rsquared(t, data_shape, feature_rdm, shm, its=10):

    print(f'Calculating r^2 t= {t}')

    existing_shm = shared_memory.SharedMemory(name=shm)
    eeg_rdms = np.ndarray(data_shape, dtype='float64', buffer=existing_shm.buf)

    tic = time.time()

    # Calc correlation
    eeg_rdm = eeg_rdms[:, :, t]
    rdm_cor = spearmanr(squareform(eeg_rdm, checks=False), squareform(feature_rdm, checks=False))[0] 

    # Calc significance
    rdm_cor_p = corr_nullDist(rdm_cor, eeg_rdm, feature_rdm, its=its) # testing purpose

    toc = time.time()

    print(f'iteration {t} in {toc-tic}')

    return (rdm_cor, rdm_cor_p)


def cor_variability(t, data_shape, feature_rdm, shm, its=100):

    print(f'Calculating CI t= {t}')

    existing_shm = shared_memory.SharedMemory(name=shm)
    eeg_rdms = np.ndarray(data_shape, dtype='float64', buffer=existing_shm.buf)
    eeg_rdm = eeg_rdms[:, :, t]

    tic = time.time()

    rdm_corr_boots = []

    for i in range(its):

        if i % 100 == 0:
            print(f'cor var iteration: {i}')

        # Create a random index that respects the structure of an rdm.
        sample = np.random.choice(eeg_rdm.shape[0], eeg_rdm.shape[0],replace=True) 

        # Subsample from both the reference and the feature RDM
        eeg_rdm_sample = eeg_rdm[sample] 
        feature_rdm_sample = feature_rdm[sample] 

        # correlating with neural similarty matrix
        rdm_corr_boots.append(spearmanr(squareform(eeg_rdm_sample, checks=False), squareform(feature_rdm_sample, checks=False))[0])

    # Get 95% confidence interval
    lower_p = np.percentile(rdm_corr_boots, 2.5)
    upper_p = np.percentile(rdm_corr_boots, 97.5)

    toc = time.time()
    print(f'iteration {t} in {toc-tic}')

    return (lower_p, upper_p)


def calc_rsquared_rw(t, data_shape, design_matrix, shm, its=10):

    print(f'Calculating r^2 t= {t}')

    existing_shm = shared_memory.SharedMemory(name=shm)
    eeg_rdms = np.ndarray(data_shape, dtype='float64', buffer=existing_shm.buf)

    tic = time.time()

    eeg_rdm = eeg_rdms[:, :, t]
    sq_eeg_rdm  = squareform(eeg_rdm, checks=False)

    # Reweighting of model rdms using multiple linear regression
    model = LinearRegression().fit(design_matrix, sq_eeg_rdm)
    coefs = model.coef_

    # rw_rdms = 0
    # for i in range(design_matrix.shape[1]-1):
    #     sq_rdm = design_matrix[:, i]
    #     rw_rdm = coefs[i] * sq_rdm
    #     rw_rdms = rw_rdms + rw_rdm
    # rw_rdms = rw_rdms + coefs[-1]

    # # Calculating cor & p
    # rdm_cor = spearmanr(sq_eeg_rdm, rw_rdms)[0] 
    # rdm_cor_p = corr_nullDist(rdm_cor, eeg_rdm, squareform(rw_rdms), its=its) 

    # toc = time.time()

    # print(f'iteration {t} in {toc-tic}')

    # return (rdm_cor, rdm_cor_p)

    # Predict 
    r2_score = model.score(design_matrix, sq_eeg_rdm)
    rdm_cor_p = np.nan
    return (r2_score, rdm_cor_p)
    
