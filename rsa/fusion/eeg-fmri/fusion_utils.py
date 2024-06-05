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
 

def calc_rsquared(t, roi_rdm, data_shape, shm, dtype, eval_method='spearman'):
    """ Calculates similarity between EEG and fRMI RDM.

    Parameters
    ----------
    t: int
        Time point
    roi_rdm : float array
        fMRI data array (conditions, conditions)
    data_shape: array shape
        Shape of eeg array
    shm: str
        Name for shared memory
    eval_method: str
        Method used to calculate similarity between RDMs; 'spearman' or 'pearson'


    Returns
    -------
    rdm_cor: float
       Correlation value between EEG and fMRI RDM

    """

    existing_shm = shared_memory.SharedMemory(name=shm)
    eeg_rdms = np.ndarray(data_shape, dtype=dtype, buffer=existing_shm.buf)

    tic = time.time()

    # Calc correlation
    eeg_rdm = eeg_rdms[:, :, t]

    if eval_method == 'spearman':
        rdm_cor = (spearmanr(squareform(eeg_rdm, checks=False), squareform(roi_rdm, checks=False))[0]**2)*100
    elif eval_method == 'pearson':
        rdm_cor = (pearsonr(squareform(eeg_rdm, checks=False), squareform(roi_rdm, checks=False))[0]**2)*100
    
    toc = time.time()

    print(f't = {t} in {toc-tic}')

    return rdm_cor

