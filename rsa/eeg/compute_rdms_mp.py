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


def compute_rdms_multi(eeg_data, pseudo_order, ts, batch, data_split='train', distance_type='euclidean', n_cpus=2, shm_name='', resampled=False):
    """ Creates shared memory and sets up parallel computing (over time points) to calculate 
    EEG RDMs.

	Parameters
	----------
	eeg_data : float array
		EEG data array (timepoints, channels, conditions)
	pseudo_order : int array
		Array with pseudo order of trials (timepoints, conditions)
	ts : int
		Number of timepoints
	batch : int
        Batch number
	data_split: str
        Train or test. 
    distance_type: str
        Distance type: euclidean, pearson, euclidean-cv, classification or dv-classification
    n_cpus: int
        Number of cpus
    shm_name: str
        Name for shared memory
    resampled: bool
        Concerning resampled test RDM y/n.  

	Returns
	-------
	list (restults): 
        List of RDMs for with length ts. 

	"""

    # Allocating array
    if data_split=='train':
        n_conditions = 1000
    elif data_split=='test':
        n_conditions = 102
    
    # Creating shared memory
    shm_name = f'{shm_name}_{data_split}_{distance_type}_shared_{batch}'

    try:
        shm = shared_memory.SharedMemory(create=True, size=eeg_data.nbytes, name=shm_name)
    except FileExistsError:
        shm_old = shared_memory.SharedMemory(shm_name, create=False)
        shm_old.close()
        shm_old.unlink()
        shm = shared_memory.SharedMemory(create=True, size=eeg_data.nbytes, name=shm_name)

    # Create a np.recarray using the buffer of shm
    shm_rdms_array = np.ndarray(shape=eeg_data.shape, dtype=eeg_data.dtype, buffer=shm.buf)

    # Copy the data into the shared memory
    shm_rdms_array[:] = eeg_data[:] 

    # Parallel calculating of RDMs for timepoints
    partial_compute_rdm = partial(compute_rdm,
                                data_shape = eeg_data.shape,
                                pseudo_order=pseudo_order,
                                shm = shm_name,
                                data_split=data_split,
                                distance_type= distance_type,
                                dtype=eeg_data.dtype,
                                resampled=resampled)

    tic = time.time()
    ts = range(eeg_data.shape[2])
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_compute_rdm, ts)
    pool.close()
    toc = time.time()

    shm.close()
    shm.unlink()

    print('permutation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))
  