import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.utils import resample
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.svm import SVC
import argparse
from functools import partial
import multiprocessing as mp
from multiprocessing import current_process, cpu_count, shared_memory
import concurrent.futures
import time
from vp_utils import *

def calc_common_mp(eeg_rdms, roi_rdm, model_rdms, feature_oi, jobarr_id, its=10, n_cpus=2, shm_name='', method='spearman'):

    # Creating shared memory
    shm_name = f'shared_data_cors_{shm_name}{jobarr_id}'

    try:
        shm = shared_memory.SharedMemory(create=True, size=eeg_rdms.nbytes, name=shm_name)
    except FileExistsError:
        shm_old = shared_memory.SharedMemory(shm_name, create=False)
        shm_old.close()
        shm_old.unlink()
        shm = shared_memory.SharedMemory(create=True, size=eeg_rdms.nbytes, name=shm_name)

    # Create a np.recarray using the buffer of shm
    shm_rdms_array = np.ndarray(shape=eeg_rdms.shape, dtype=eeg_rdms.dtype, buffer=shm.buf)

    # Copy the data into the shared memory
    shm_rdms_array[:] = eeg_rdms[:]

    # Parallel calculating of r^2 for timepoints
    partial_calc_common = partial(calc_common,
                                    data_shape=eeg_rdms.shape,
                                    dtype = eeg_rdms.dtype,
                                    model_rdms = model_rdms,
                                    feature_oi = feature_oi,
                                    roi_rdm = roi_rdm,
                                    shm = shm_name,
                                    method = method,
                                    its=its)


    tic = time.time()

    ts = range(eeg_rdms.shape[2])
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_common, ts)
    pool.close()
   
    shm.close()
    shm.unlink()
    
    toc = time.time()

    print('R^2 calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))


def calc_common_cis_mp(eeg_rdms, roi_rdm, model_rdms, feature_oi, jobarr_id, its=10, n_cpus=2, shm_name='', method='spearman'):

    # Creating shared memory
    shm_name = f'shared_data_cis_{shm_name}{jobarr_id}'

    try:
        shm = shared_memory.SharedMemory(create=True, size=eeg_rdms.nbytes, name=shm_name)
    except FileExistsError:
        shm_old = shared_memory.SharedMemory(shm_name, create=False)
        shm_old.close()
        shm_old.unlink()
        shm = shared_memory.SharedMemory(create=True, size=eeg_rdms.nbytes, name=shm_name)

    # Create a np.recarray using the buffer of shm
    shm_rdms_array = np.ndarray(shape=eeg_rdms.shape, dtype=eeg_rdms.dtype, buffer=shm.buf)

    # Copy the data into the shared memory
    shm_rdms_array[:] = eeg_rdms[:]

    # Parallel calculating of r^2 for timepoints
    partial_calc_common = partial(calc_cis_common,
                                    data_shape=eeg_rdms.shape,
                                    dtype = eeg_rdms.dtype,
                                    model_rdms = model_rdms,
                                    feature_oi = feature_oi,
                                    roi_rdm = roi_rdm,
                                    shm = shm_name,
                                    method = method,
                                    its=its)


    tic = time.time()

    ts = range(eeg_rdms.shape[2])
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_common, ts)
    pool.close()
   
    shm.close()
    shm.unlink()
    
    toc = time.time()

    print('R^2 calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))