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
from create_rdms_utils import compute_rdm
import concurrent.futures
import time


def compute_rdms_multi(eeg_data, pseudo_order, ts, batch, data_split='train', distance_type='euclidean', n_cpus=8):

    # Allocating array
    if data_split=='train':
        n_conditions = 1000
    elif data_split=='test':
        n_conditions == 102
    
    # rdms_array = np.zeros((n_conditions, n_conditions, eeg_data.shape[2]), dtype='float32')

    # Creating shared memory
    shm_name = f'shared_data_permute_{batch}'

    try:
        shm = shared_memory.SharedMemory(create=True, size=eeg_data.nbytes, name=shm_name)
    except FileExistsError:
        shm_old = shared_memory.SharedMemory(shm_name, create=False)
        shm_old.close()
        shm_old.unlink()
        shm = shared_memory.SharedMemory(create=True, size=eeg_data.nbytes, name=shm_name)

    # Create a np.recarray using the buffer of shm
    shm_rdms_array = np.recarray(shape=eeg_data.shape, dtype=eeg_data.dtype, buf=shm.buf)

    # Copy the data into the shared memory
    np.copyto(shm_rdms_array, eeg_data)

    # Parallel calculating of RDMs for timepoints
    partial_compute_rdm = partial(compute_rdm,
                                data_shape = eeg_data.shape,
                                pseudo_order=pseudo_order,
                                shm = shm_name,
                                data_split=data_split,
                                distance_type= distance_type)

    tic = time.time()
    ts = range(eeg_data.shape[2])
    pool = mp.Pool(n_cpus)
    # with concurrent.futures.ProcessPoolExecutor(cpu_count()) as executor:
    #     ts = range(eeg_data.shape[2])
    #     results = executor.map(partial_compute_rdm, ts)
    results = pool.map(partial_compute_rdm, ts)
    pool.close()
    toc = time.time()

    shm.close()
    shm.unlink()

    print('permutation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))



 