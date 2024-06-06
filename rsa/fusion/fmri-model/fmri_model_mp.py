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


def calc_cor_mp(rois, fmri_data, feature_rdm, its=10, n_cpus=2, eval_method='spearman', booted=False):
    """ Sets up parallel computing (over different rois) to calculate similarity between 
    fMRI RDM and model RDM. Subject level.

    Parameters
    ----------
    rois: list of str
        List with roi names
    fmri_data : dictionary with arrays
        Dictionary with arrays for ROIs. 
    feature_rdm: array of floats
        Model RDM. 
    its: int
        n of iterations used to compute significance
    n_cpus: int
        Number of cpus
    eval_method: str
        Method used to calculate similarity between RDMs; 'spearman' or 'pearson'
    booted: bool
        To ue bootstrapped-based significance calculation y/n


    Returns
    -------
    list (results): 
        List of correlation results for all ROIs.

    """

    # Parallel calculating of r^2 for timepoints
    if booted == False:
        partial_calc_cor = partial(calc_corr,
                                        fmri_data = fmri_data,
                                        feature_rdm = feature_rdm,
                                        eval_method = eval_method,
                                        its=its)
    else:
        partial_calc_cor = partial(calc_corr_boot,
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


def calc_cor_ga_mp(rois, fmri_data, feature_rdm, noise_ceilings, its=10, n_cpus=2, eval_method='spearman', booted=False):
    """ Sets up parallel computing (over different rois) to calculate similarity between 
    fMRI RDM and model RDM. Group level --> uses different computing function that includes normalization using noise ceiling.

    Parameters
    ----------
    rois: list of str
        List with roi names
    fmri_data : dictionary with arrays
        Dictionary with arrays for ROIs. 
    feature_rdm: array of floats
        Model RDM. 
    its: int
        n of iterations used to compute significance
    n_cpus: int
        Number of cpus
    eval_method: str
        Method used to calculate similarity between RDMs; 'spearman' or 'pearson'
    booted: bool
        To ue bootstrapped-based significance calculation y/n


    Returns
    -------
    list (results): 
        List of correlation results for all ROIs (correlation, p-value).

    """

    if booted == False:
        partial_calc_cor = partial(calc_corr_GA,
                                fmri_data = fmri_data,
                                feature_rdm = feature_rdm,
                                eval_method = eval_method,
                                its=its,
                                noise_ceilings=noise_ceilings)
    else: 
        partial_calc_cor = partial(calc_corr_GA_boot,
                        fmri_data = fmri_data,
                        feature_rdm = feature_rdm,
                        eval_method = eval_method,
                        its=its,
                        noise_ceilings=noise_ceilings)

    tic = time.time()
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_cor, rois)
    pool.close()
    toc = time.time()

    print('Cor calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))


def calc_cis_mp(rois, fmri_data, feature_rdm, its=10, n_cpus=2, eval_method='spearman'):
    """ Sets up parallel computing (over different rois) to calculate 95 % confidence interval between 
    fMRI RDM and model RDM. Subject level.

    Parameters
    ----------
    rois: list of str
        List with roi names
    fmri_data : dictionary with arrays
        Dictionary with arrays for ROIs. 
    feature_rdm: array of floats
        Model RDM. 
    its: int
        n of iterations used to compute significance
    n_cpus: int
        Number of cpus
    eval_method: str
        Method used to calculate similarity between RDMs; 'spearman' or 'pearson'

    Returns
    -------
    list (results): 
        List of cis results for all ROIs (lower ci, upper ci).

    """

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


def calc_cis_GA_mp(rois, fmri_data, feature_rdm, noise_ceilings, its=10, n_cpus=2, eval_method='spearman'):
    """ Sets up parallel computing (over different rois) to calculate 95 % confidence interval between 
    fMRI RDM and model RDM. Subject level.  Group level --> uses different computing function that includes normalization using noise ceiling.

    Parameters
    ----------
    rois: list of str
        List with roi names
    fmri_data : dictionary with arrays
        Dictionary with arrays for ROIs. 
    feature_rdm: array of floats
        Model RDM. 
    its: int
        n of iterations used to compute significance
    n_cpus: int
        Number of cpus
    eval_method: str
        Method used to calculate similarity between RDMs; 'spearman' or 'pearson'

    Returns
    -------
    list (results): 
        List of cis results for all ROIs (lower ci, upper ci).

    """

    # Parallel calculating of r^2 for timepoints
    partial_calc_cis = partial(cor_variability_GA,
                                    fmri_data = fmri_data,
                                    feature_rdm = feature_rdm,
                                    eval_method = eval_method,
                                    noise_ceilings=noise_ceilings,
                                    its=its)
    
    tic = time.time()
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_cis, rois)
    pool.close()
    toc = time.time()
    
    toc = time.time()

    print('CI calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))



def calc_rsquared_rw_mp(rois, fmri_data, design_matrices, its, n_cpus=2, inspect=False):
    """ Sets up parallel computing (over different rois) to perform variance partitioning for
    fMRI RDM and model RDMs. Group level.

    Parameters
    ----------
    rois: list of str
        List with roi names
    fmri_data : dictionary with arrays
        Dictionary with arrays for ROIs. 
    design matrices: array of floats
        Combination of flattend model RDMs.
    its: int
        n of iterations used to compute significance
    n_cpus: int
        Number of cpus
    inspect: bool
        Make inspection plots y/n

    Returns
    -------
    list (results): 
        List of correlation results for all ROIs (variance explained, p-value).

    """

    # Parallel calculating of r^2 for timepoints
    partial_calc_rsquared = partial(calc_rsquared_rw,
                                    fmri_data=fmri_data,
                                    design_matrices = design_matrices,
                                    its=its,
                                    inspect=inspect)


    tic = time.time()
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_rsquared, rois)
    pool.close() 
    toc = time.time()

    print('R^2 calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))


def calc_cis_rw_mp(rois, fmri_data, design_matrices, its, n_cpus=2):
    """ Sets up parallel computing (over different rois) to perform variance partitioning for
    fMRI RDM and model RDMs. Group level.

    --> unreliable results.

    Parameters
    ----------
    rois: list of str
        List with roi names
    fmri_data : dictionary with arrays
        Dictionary with arrays for ROIs. 
    design matrices: array of floats
        Combination of flattend model RDMs.
    its: int
        n of iterations used to compute significance
    n_cpus: int
        Number of cpus
 
    Returns
    -------
    list (results): 
        List of cis results for all ROIs (lower ci, upper ci).

    """

    # Parallel calculating of r^2 for timepoints
    partial_calc_cis = partial(calc_variability_rw,
                                    fmri_data=fmri_data,
                                    design_matrices = design_matrices,
                                    its=its)


    tic = time.time()
    pool = mp.Pool(n_cpus)
    results = pool.map(partial_calc_cis, rois)
    pool.close() 
    toc = time.time()

    print('R^2 calculation done in {:.4f} seconds'.format(toc-tic))

    return(list(results))