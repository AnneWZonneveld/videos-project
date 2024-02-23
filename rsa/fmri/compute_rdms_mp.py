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
from multiprocessing import current_process, cpu_count, Manager
from create_rdms_utils import compute_rdm
import concurrent.futures
import time
import copy


def compute_rdms_multi(fmri_data, pseudo_order, batch, shm_name, n_cpus, data_split='test', distance_type='pearson'):

    # Parallel calculating of RDMs for rois
    partial_compute_rdm = partial(compute_rdm,
                                fmri_data = fmri_data,
                                pseudo_order= pseudo_order,
                                shm_name = shm_name,
                                data_split=data_split,
                                distance_type= distance_type)

    tic = time.time()
    rois = fmri_data.keys()
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_compute_rdm, rois)
    pool.close()
    toc = time.time()

    return(list(results))