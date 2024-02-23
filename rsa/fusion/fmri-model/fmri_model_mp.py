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
from fmri_model_utils import *


def calc_cor_mp(rois, fmri_data, feature_rdm, its=10, n_cpus=2, eval_method='spearman'):

    # Parallel calculating of r^2 for timepoints
    partial_calc_cor = partial(calc_corr,
                                    fmri_data = fmri_data,
                                    feature_rdm = feature_rdm,
                                    eval_method = eval_method,
                                    its=its)


    tic = time.time()
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_cor, rois)
    pool.close()
    toc = time.time()

    print('Cor calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))


def calc_cis_mp(rois, fmri_data, feature_rdm, its=10, n_cpus=2, eval_method='spearman'):

    # Parallel calculating of r^2 for timepoints
    partial_calc_cis = partial(cor_variability,
                                    fmri_data = fmri_data,
                                    feature_rdm = feature_rdm,
                                    eval_method = eval_method,
                                    its=its)
    
    tic = time.time()
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_cis, rois)
    pool.close()
    toc = time.time()
    
    toc = time.time()

    print('CI calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))


def calc_rsquared_rw_mp(rois, fmri_data, design_matrices, its, n_cpus=2):

    # Parallel calculating of r^2 for timepoints
    partial_calc_rsquared = partial(calc_rsquared_rw,
                                    fmri_data=fmri_data,
                                    design_matrices = design_matrices,
                                    its=its)


    tic = time.time()
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_rsquared, rois)
    pool.close() 
    toc = time.time()

    print('R^2 calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))