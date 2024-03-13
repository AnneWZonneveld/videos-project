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
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import fdrcorrection
import argparse
import pingouin as pg
import random
import time
from multiprocessing import shared_memory


def common_nullDist(commonality, eeg_rdm, roi_rdm, model_rdms, feature_oi, its=10000, method='spearman'):

    object_model = model_rdms['objects']
    scene_model = model_rdms['scenes']
    action_model = model_rdms['actions']

    rdm_common_null = []
    for i in range(its):

        if i % 1000 == 0:
            print(f'null distr iteration: {i}')

        # Create a random index that respects the structure of an rdm.
        random.seed(i)
        shuffle = np.random.choice(eeg_rdm.shape[0], eeg_rdm.shape[0], replace=False)

        # shuffle RDM consistently for both dims
        shuffled_eeg_rdm = eeg_rdm[shuffle,:] # rows
        shuffled_eeg_rdm = eeg_rdm[:,shuffle] # columns

        df = pd.DataFrame()
        df['eeg'] = squareform(shuffled_eeg_rdm, checks=False)
        df['fmri'] = squareform(roi_rdm, checks=False)
        df['objects'] = squareform(object_model, checks=False)
        df['scenes'] = squareform(scene_model, checks=False)
        df['actions'] = squareform(action_model, checks=False)

        vars = ['objects', 'scenes', 'actions']
        y_covar = [i for i in vars if i != feature_oi]

        # EG looking at objects (semi partial cor)
        Q1 = pg.partial_corr(data=df, x='eeg', y='fmri', y_covar=y_covar, method=method)
        Q2 = pg.partial_corr(data=df, x='eeg', y='fmri', y_covar=vars, method=method)
        commonality = (Q1['r'][0]**2 - Q2['r'][0]**2)*100

        rdm_common_null.append(commonality)

    # Calc p value 
    rdm_common_null = np.array(rdm_common_null)
    p_val = np.mean(rdm_common_null>commonality)

    return p_val
    

def calc_common(t, roi_rdm, model_rdms, feature_oi, data_shape, shm, dtype, its=10000, method='spearman'):

    existing_shm = shared_memory.SharedMemory(name=shm)
    eeg_rdms = np.ndarray(data_shape, dtype=dtype, buffer=existing_shm.buf)

    tic = time.time()

    # Calc correlation
    eeg_rdm = eeg_rdms[:, :, t]
    object_model = model_rdms['objects']
    scene_model = model_rdms['scenes']
    action_model = model_rdms['actions']

    df = pd.DataFrame()
    df['eeg'] = squareform(eeg_rdm, checks=False)
    df['fmri'] = squareform(roi_rdm, checks=False)
    df['objects'] = squareform(object_model, checks=False)
    df['scenes'] = squareform(scene_model, checks=False)
    df['actions'] = squareform(action_model, checks=False)

    vars = ['objects', 'scenes', 'actions']
    y_covar = [i for i in vars if i != feature_oi]

    # EG looking at objects (semi partial cor)
    Q1 = pg.partial_corr(data=df, x='eeg', y='fmri', y_covar=y_covar, method=method)
    Q2 = pg.partial_corr(data=df, x='eeg', y='fmri', y_covar=vars, method=method)
    commonality = (Q1['r'][0]**2 - Q2['r'][0]**2)*100

    common_p = common_nullDist(commonality=commonality, eeg_rdm=eeg_rdm, roi_rdm=roi_rdm, model_rdms=model_rdms, feature_oi=feature_oi, its=its, method=method) 

    toc = time.time()

    return (commonality, common_p)



def calc_cis_common(t, roi_rdm, model_rdms, feature_oi, data_shape, shm, dtype, its=10000, method='spearman'):

    tic = time.time()
    existing_shm = shared_memory.SharedMemory(name=shm)
    eeg_rdms = np.ndarray(data_shape, dtype=dtype, buffer=existing_shm.buf)
    eeg_rdm = eeg_rdms[:, :, t]
    sq_eeg_rdm = squareform(eeg_rdm, checks=False)

    object_model = model_rdms['objects']
    scene_model = model_rdms['scenes']
    action_model = model_rdms['actions']

    rdm_common_boots = []
    for i in range(its):

        if i % 1000 == 0:
            print(f'null distr iteration: {i}')

        # Create a random index that respects the structure of an rdm.
        random.seed(i)
        sample = np.random.choice(sq_eeg_rdm.shape[0], sq_eeg_rdm.shape[0], replace=False)

        df = pd.DataFrame()
        df['eeg'] = sq_eeg_rdm[sample]
        df['fmri'] = squareform(roi_rdm, checks=False)[sample]
        df['objects'] = squareform(object_model, checks=False)[sample]
        df['scenes'] = squareform(scene_model, checks=False)[sample]
        df['actions'] = squareform(action_model, checks=False)[sample]

        vars = ['objects', 'scenes', 'actions']
        y_covar = [i for i in vars if i != feature_oi]

        # EG looking at objects (semi partial cor)
        Q1 = pg.partial_corr(data=df, x='eeg', y='fmri', y_covar=y_covar, method=method)
        Q2 = pg.partial_corr(data=df, x='eeg', y='fmri', y_covar=vars, method=method)
        commonality = (Q1['r'][0]**2 - Q2['r'][0]**2)*100

        rdm_common_boots.append(commonality)

    # Calc cis
    lower_cis =  np.percentile(rdm_common_boots, 2.5)
    upper_cis = np.percentile(rdm_common_boots, 97.5)

    toc = time.time()

    return (lower_cis, upper_cis)