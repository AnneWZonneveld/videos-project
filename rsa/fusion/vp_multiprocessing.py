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

def calc_rsquared_mp(GA_rdms, feature_rdm, feature, its=10, n_cpus=8):

    # Creating shared memory
    shm_name = f'shared_data_permute_{feature}'

    try:
        shm = shared_memory.SharedMemory(create=True, size=GA_rdms.nbytes, name=shm_name)
    except FileExistsError:
        shm_old = shared_memory.SharedMemory(shm_name, create=False)
        shm_old.close()
        shm_old.unlink()
        shm = shared_memory.SharedMemory(create=True, size=GA_rdms.nbytes, name=shm_name)

    # Create a np.recarray using the buffer of shm
    shm_rdms_array = np.recarray(shape=GA_rdms.shape, dtype=GA_rdms.dtype, buf=shm.buf)

    # Copy the data into the shared memory
    np.copyto(shm_rdms_array, GA_rdms)

    # Parallel calculating of r^2 for timepoints
    partial_calc_rsquared = partial(calc_rsquared,
                                    data_shape=GA_rdms.shape,
                                    feature_rdm = feature_rdm,
                                    shm = shm_name,
                                    its=10)


    tic = time.time()
    ts = range(GA_rdms.shape[2])
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_rsquared, ts)
    pool.close()
    toc = time.time()

    shm.close()
    shm.unlink()

    print('R^2 calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))