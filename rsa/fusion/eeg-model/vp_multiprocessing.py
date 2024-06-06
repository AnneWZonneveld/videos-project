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

def calc_rsquared_mp(eeg_rdms, feature_rdm, jobarr_id, its=10, n_cpus=2, shm_name='', eval_method='spearman'):
    """ Creates shared memory and sets up parallel computing (over different time points) to calculate similarity between 
    EEG and model RDMS.

    Parameters
    ----------
    eeg_rdms: float array
        eeg data array (time points, conditions, conditions)
    feature_rdm : float array
        model rdm array (conditions, conditions)
    jobarr_id: int
        unique job id
    its: int
        n of iterations used to compute significance
    n_cpus: int
        Number of cpus
    shm_name: str
        Name for shared memory
    eval_method: str
        Method used to calculate similarity between RDMs; 'spearman' or 'pearson'


    Returns
    -------
    list (restults): 
        List of correlation results (cor value, p-value).

    """

    print('Calc_rsquared_mp')

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
    print(f'eeg rdms {eeg_rdms[1,5,1]}')
    print(f'shared memory {shm_rdms_array.shape}, {shm_rdms_array[1, 5,1]}')

    # Parallel calculating of r^2 for timepoints
    partial_calc_rsquared = partial(calc_rsquared,
                                    data_shape=eeg_rdms.shape,
                                    dtype = eeg_rdms.dtype,
                                    feature_rdm = feature_rdm,
                                    shm = shm_name,
                                    eval_method = eval_method,
                                    its=its)


    tic = time.time()
    print('opening pool')
    ts = range(eeg_rdms.shape[2])
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_rsquared, ts)
    pool.close()
   
    shm.close()
    shm.unlink()
    
    toc = time.time()

    print('R^2 calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))


def calc_cis_mp(eeg_rdms, feature_rdm, jobarr_id, feature='objects', distance_type='pearson', data_split='train', its=10, n_cpus=2, shm_name='', eval_method='spearman'):
    """ Creates shared memory and sets up parallel computing (over different time points) to calculate 95 % confidence interval between 
    EEG and model RDMs.

    Parameters
    ----------
    eeg_rdms: float array
        eeg data array (time points, conditions, conditions)
    feature_rdm : float array
        model rdm array (conditions, conditions)
    jobarr_id: int
        unique job id
    its: int
        n of iterations used to compute significance
    n_cpus: int
        Number of cpus
    shm_name: str
        Name for shared memory
    eval_method: str
        Method used to calculate similarity between RDMs; 'spearman' or 'pearson'


    Returns
    -------
    list (restults): 
        List of correlation results (lower ci, upper ci).

    """

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
    partial_calc_cis = partial(cor_variability,
                                    data_shape=eeg_rdms.shape,
                                    dtype = eeg_rdms.dtype,
                                    feature_rdm = feature_rdm,
                                    distance_type = distance_type,
                                    data_split = data_split,
                                    shm = shm_name,
                                    eval_method=eval_method,
                                    feature=feature,
                                    its=its)
    
    ts = range(eeg_rdms.shape[2])
    tic = time.time()
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_cis, ts)
    pool.close()
   
    shm.close()
    shm.unlink()
    
    toc = time.time()

    print('CI calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))


def calc_rsquared_rw_mp(eeg_rdms, design_matrix, jobarr_id, n_cpus=2, ridge=0, cv=0, shm_name=''):
    """ Outdated function """

    # Creating shared memory
    shm_name = f'shared_data_rw_cors_{shm_name}{jobarr_id}'

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
    partial_calc_rsquared = partial(calc_rsquared_rw,
                                    data_shape=eeg_rdms.shape,
                                    dtype = eeg_rdms.dtype,
                                    design_matrix = design_matrix,
                                    shm = shm_name,
                                    ridge=ridge,
                                    cv=cv)


    tic = time.time()

    ts = range(eeg_rdms.shape[2])
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_rsquared, ts)
    pool.close()
   
    shm.close()
    shm.unlink()
    
    toc = time.time()

    print('R^2 calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))


def calc_rsquared_rw2_mp(eeg_rdms, design_matrices, jobarr_id, its, n_cpus=2, shm_name=''):
    """ Creates shared memory and sets up parallel computing (over different time points) to perform variance partitioning between 
    EEG and model RDMs.

    Parameters
    ----------
    eeg_rdms: float array
        eeg data array (time points, conditions, conditions)
    design_matrices : float array
        combination of flattened model rdms of interest
    jobarr_id: int
        unique job id
    its: int
        n of iterations used to compute significance
    n_cpus: int
        Number of cpus
    shm_name: str
        Name for shared memory

    Returns
    -------
    list (restults): 
        List of correlation results (variance explained, p-value).

    """


    # Creating shared memory
    shm_name = f'shared_data_rw2_{shm_name}{jobarr_id}'

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
    partial_calc_rsquared = partial(calc_rsquared_rw_2,
                                    data_shape=eeg_rdms.shape,
                                    dtype= eeg_rdms.dtype,
                                    design_matrices = design_matrices,
                                    shm = shm_name,
                                    its=its)


    tic = time.time()

    ts = range(eeg_rdms.shape[2])
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_rsquared, ts)
    pool.close()
   
    shm.close()
    shm.unlink()
    
    toc = time.time()

    print('R^2 calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))


def calc_cis_rw2_mp(eeg_rdms, design_matrices, jobarr_id, distance_type='pearson', data_split='train', its=10, n_cpus=2, shm_name='', eval_method='spearman'):
    """ Creates shared memory and sets up parallel computing (over different time points) to calculate 95% confidence 
    interval for variance partitioning between EEG and model RDMs.

    ---> results unreliable.

    Parameters
    ----------
    eeg_rdms: float array
        eeg data array (time points, conditions, conditions)
    design_matrices : float array
        combination of flattened model rdms of interest
    jobarr_id: int
        unique job id
    its: int
        n of iterations used to compute significance
    n_cpus: int
        Number of cpus
    shm_name: str
        Name for shared memory

    Returns
    -------
    list (restults): 
        List of correlation results (lower ci, upper ci).

    """


    # Creating shared memory
    shm_name = f'shared_data_cis_rw2_{shm_name}{jobarr_id}'

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
    partial_calc_cis = partial(cor_variability_rw,
                                    data_shape=eeg_rdms.shape,
                                    dtype = eeg_rdms.dtype,
                                    design_matrices = design_matrices,
                                    shm = shm_name,
                                    its=its)
    
    ts = range(eeg_rdms.shape[2])
    tic = time.time()
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_cis, ts)
    pool.close()
   
    shm.close()
    shm.unlink()
    
    toc = time.time()

    print('CI calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))