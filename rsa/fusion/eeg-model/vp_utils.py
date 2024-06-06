import os
import random
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
from sklearn.metrics import pairwise_distances, r2_score
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import time
from multiprocessing import shared_memory


def corr_nullDist(rdm_cor, rdm_1, rdm_2, its=100, eval_method='spearman'):
    """ Calculates null distribution based on permutations for correlation calculation. 

    Parameters
    ----------
    rdm_cor: float 
        Found experimental correlation value
    rdm_1 : float array
        RDM 1 (= EEG RDM) (conditions, conditions)
    rdm_2 : float array
        RDM 2 (= model RDM) (conditions, conditions)
    its: int
        n of iterations used to compute significance
    eval_method: str
        Evaluation method to calculate similarity; 'spearman'/'pearson'

    Returns
    -------
    p-val: float
        P-value.
    """

    print("Constructing null distribution")

    rdv_1 = squareform(rdm_1, checks=False)

    rdm_corr_null = []
    for i in range(its):

        if i % 1000 == 0:
            print(f'null distr iteration: {i}')

        # Create a random index 
        random.seed(i)
        shuffle = np.random.choice(rdv_1.shape[0], rdv_1.shape[0], replace=False)

        # shuffle RDM consistently for both dims
        shuffled_rdv_1 = rdv_1[shuffle] # rows

        # correlating with neural similarty matrix
        if eval_method == 'spearman':
            rdm_corr_null.append(spearmanr(shuffled_rdv_1, squareform(rdm_2, checks=False))[0])
        elif eval_method == 'pearson':
            rdm_corr_null.append(pearsonr(shuffled_rdv_1, squareform(rdm_2, checks=False))[0])
    
    # Calc p value
    rdm_corr_null = np.array(rdm_corr_null)
    p_val = np.mean(rdm_corr_null>rdm_cor) 

    return p_val


def corr_nullDist_rw(vp_scores, design_matrices, eeg_rdm, its=100):
    """ Calculates null distribution based on permutations for variance partitioning. 

    Parameters
    ----------
    vp_scores: dict of floats
        Found experimental variance explained values.
    design_matrice: float array
        Combination of flattened model rdms of interest. 
    eeg_rdm: float array
        EEG rdm (conditions, conditions)
    its: int
        n of iterations used to compute significance

    Returns
    -------
    p-val: float
        P-value.
    """

    print("Constructing null distribution")

    sq_eeg_rdm = squareform(eeg_rdm, checks=False)
    
    vp_perm_scores = {
            'u_a': [],
            'u_s': [],
            'u_o': [],
            'os_shared' : [],
            'sa_shared' : [],
            'oa_shared' : [],
            'osa_shared'  : [],
            'o_total' : [],
            's_total': [], 
            'a_total' : [],
            'all_parts': []
            }
    
    for i in range(its):

        if i % 1000 == 0:
            print(f'null distr iteration: {i}')

        # Create a random index that respects the structure of an rdm.
        random.seed(i)
        shuffle = np.random.choice(sq_eeg_rdm.shape[0], sq_eeg_rdm.shape[0],replace=False)
        shuffled_eeg = sq_eeg_rdm[shuffle]
        
        # Calc r2 for different models
        r2_scores = {}
        models = design_matrices.keys()
        for model in models:
            design_matrix = design_matrices[model]
            model_fit = LinearRegression().fit(design_matrix, shuffled_eeg)
            r2 = model_fit.score(design_matrix, shuffled_eeg)
            r2_scores[model] = r2*100
        
        # Perform variance partitioning 
        u_a_r2 = r2_scores['o-s-a'] - r2_scores['o-s']
        u_s_r2 = r2_scores['o-s-a'] - r2_scores['o-a']
        u_o_r2 = r2_scores['o-s-a'] - r2_scores['s-a']

        os_shared_r2 = r2_scores['o-s-a'] - r2_scores['a'] - u_o_r2 - u_s_r2
        sa_shared_r2 = r2_scores['o-s-a'] - r2_scores['o'] - u_s_r2 - u_a_r2
        oa_shared_r2 = r2_scores['o-s-a'] - r2_scores['s'] - u_o_r2 - u_a_r2

        osa_shared_r2 = r2_scores['o-s-a'] - u_o_r2 - u_s_r2 - u_a_r2 - os_shared_r2 - sa_shared_r2 - oa_shared_r2

        o_total_r2 = u_o_r2 +  os_shared_r2 + oa_shared_r2 + osa_shared_r2
        s_total_r2 = u_s_r2 + os_shared_r2 + sa_shared_r2 + osa_shared_r2
        a_total_r2 = u_a_r2 + oa_shared_r2 + sa_shared_r2 + osa_shared_r2

        all_parts = u_o_r2 + u_s_r2 + u_a_r2 + os_shared_r2 + sa_shared_r2 + oa_shared_r2 + osa_shared_r2

        vp_perm_scores['u_a'].append(u_a_r2)
        vp_perm_scores['u_s'].append(u_s_r2)
        vp_perm_scores['u_o'].append(u_o_r2)
        vp_perm_scores['os_shared'].append(os_shared_r2)
        vp_perm_scores['sa_shared'].append(sa_shared_r2)
        vp_perm_scores['oa_shared'].append(oa_shared_r2)
        vp_perm_scores['osa_shared'].append(osa_shared_r2)
        vp_perm_scores['o_total'].append(o_total_r2)
        vp_perm_scores['s_total'].append(s_total_r2)
        vp_perm_scores['a_total'].append(a_total_r2)
        vp_perm_scores['all_parts'].append(all_parts)


    # Calc p value 
    p_vals= {}
    p_vals['u_a'] = np.mean(np.array(vp_perm_scores['u_a']) > vp_scores['u_a'])
    p_vals['u_s'] = np.mean(np.array(vp_perm_scores['u_s']) > vp_scores['u_s'])
    p_vals['u_o'] = np.mean(np.array(vp_perm_scores['u_o']) > vp_scores['u_o'])
    p_vals['os_shared'] = np.mean(np.array(vp_perm_scores['os_shared']) > vp_scores['os_shared'])
    p_vals['sa_shared'] = np.mean(np.array(vp_perm_scores['sa_shared']) > vp_scores['sa_shared'])
    p_vals['oa_shared'] = np.mean(np.array(vp_perm_scores['oa_shared']) > vp_scores['oa_shared'])
    p_vals['osa_shared'] = np.mean(np.array(vp_perm_scores['osa_shared']) > vp_scores['osa_shared'])
    p_vals['o_total'] = np.mean(np.array(vp_perm_scores['o_total']) > vp_scores['o_total'])
    p_vals['s_total'] = np.mean(np.array(vp_perm_scores['s_total']) > vp_scores['s_total'])
    p_vals['a_total'] = np.mean(np.array(vp_perm_scores['a_total']) > vp_scores['a_total'])
    p_vals['all_parts'] = np.mean(np.array(vp_perm_scores['all_parts']) > vp_scores['all_parts'])
    
    return p_vals


def calc_rsquared(t, data_shape, dtype, feature_rdm, shm, its=10, eval_method='spearman'):
    """ Calculates similarity between eeg rdm and model rdm at time point t. 

    Parameters
    ----------
    t: int
        Time point of interest
    data_shape: shape
        Data shape of EEG data
    dtype: type
        Data type of EEG data
    feature_rdm: array of floats (conditions, conditions)
        Model RDM
    shm: str
        Name of shared memory.
    its: int
        n of iterations used to compute significance
    eval_emthod: str
        Evulation method to calculate similarity; 'spearman'/'pearson'

    Returns
    -------
    (rdm_cor, p-val): (float, float)
        Correlation value, p-values
    """

    print(f'Calculating r^2 t= {t}')

    existing_shm = shared_memory.SharedMemory(name=shm)
    eeg_rdms = np.ndarray(data_shape, dtype=dtype, buffer=existing_shm.buf)

    tic = time.time()

    # Calc correlation
    eeg_rdm = eeg_rdms[:, :, t]

    if eval_method == 'spearman':
        rdm_cor = spearmanr(squareform(eeg_rdm, checks=False), squareform(feature_rdm, checks=False))[0] 
    elif eval_method == 'pearson':
        rdm_cor = pearsonr(squareform(eeg_rdm, checks=False), squareform(feature_rdm, checks=False))[0] 
    elif eval_method == 'euclidean': 
        rdm_cor = np.linalg.norm(squareform(eeg_rdm, checks=False) - squareform(feature_rdm, checks=False))
    
    rdm_cor_p = corr_nullDist(rdm_cor, eeg_rdm, feature_rdm, its=its, eval_method=eval_method) 

    toc = time.time()

    print(f'iteration {t} in {toc-tic}')

    return (rdm_cor, rdm_cor_p)


def cor_variability(t, data_shape, dtype, feature_rdm, shm, its=100, eval_method='spearman'):
    """ Calculates 95% confidence interval for similarity between eeg rdm and model rdm at time point t. 

    Parameters
    ----------
    t: int
        Time point of interest
    data_shape: shape
        Data shape of EEG data
    dtype: type
        Data type of EEG data
    feature_rdm: array of floats (conditions, conditions)
        Model RDM
    shm: str
        Name of shared memory.
    its: int
        n of iterations used to compute significance
    eval_emthod: str
        Evulation method to calculate similarity; 'spearman'/'pearson'

    Returns
    -------
    (lower_p, upper_p): (float, float)
        Lower percentile, upper percentile
    """

    print(f'Calculating CI t= {t}')

    existing_shm = shared_memory.SharedMemory(name=shm)
    eeg_rdms = np.ndarray(data_shape, dtype=dtype, buffer=existing_shm.buf)
    eeg_rdm = eeg_rdms[:, :, t]

    eeg_rdm_vec = squareform(eeg_rdm, checks=False)
    feature_rdm_vec = squareform(feature_rdm, checks=False)

    tic = time.time()

    rdm_corr_boots = []

    for i in range(its):

        if i % 100 == 0:
            print(f'cor var iteration: {i}')

        # Create a random index that respects the structure of an rdm.
        random.seed(i)
        sample = np.random.choice(eeg_rdm_vec.shape[0], eeg_rdm_vec.shape[0],replace=True) 

        # Subsample from both the reference and the feature RDM
        eeg_rdm_sample = eeg_rdm_vec[sample] 
        feature_rdm_sample = feature_rdm_vec[sample] 

        # correlating with neural similarty matrix
        if eval_method == 'spearman':
            rdm_corr_boots.append(spearmanr(eeg_rdm_sample, feature_rdm_sample)[0])
        elif eval_method == 'pearson':
            rdm_corr_boots.append(pearsonr(eeg_rdm_sample, feature_rdm_sample)[0])
        elif eval_method == 'euclidean':
            rdm_corr_boots.append(np.linalg.norm(eeg_rdm_sample - feature_rdm_sample))

    # Get 95% confidence interval
    lower_p = np.percentile(rdm_corr_boots, 2.5)
    upper_p = np.percentile(rdm_corr_boots, 97.5)

    toc = time.time()
    print(f'iteration {t} in {toc-tic}')

    return (lower_p, upper_p)


def cor_variability_rw(t, data_shape, dtype, design_matrices, shm, its=100):
    """ Calculates 95% confidence interval for similarity between eeg rdm and model rdm at time point t. 
    
    ---> unreliable results

    Parameters
    ----------
    t: int
        Time point of interest
    data_shape: shape
        Data shape of EEG data
    dtype: type
        Data type of EEG data
    design_matrices: array of floats 
        Combination of flattened model rdms of interest
    shm: str
        Name of shared memory.
    its: int
        n of iterations used to compute significance

    Returns
    -------
    (lower_p, upper_p): (float, float)
        Lower percentile, upper percentile
    """

    print(f'Calculating CI t= {t}')

    existing_shm = shared_memory.SharedMemory(name=shm)
    eeg_rdms = np.ndarray(data_shape, dtype=dtype, buffer=existing_shm.buf)
    eeg_rdm = eeg_rdms[:, :, t]
    eeg_rdm_vec = squareform(eeg_rdm, checks=False)
    
    tic = time.time()

    vp_boot_scores = {
        'u_a': [],
        'u_s': [],
        'u_o': [],
        'os_shared' : [],
        'sa_shared' : [],
        'oa_shared' : [],
        'osa_shared'  : [],
        'o_total' : [],
        's_total': [], 
        'a_total' : [],
        'all_parts': []
        }

    for i in range(its):

        if i % 100 == 0:
            print(f'cor var iteration: {i}')

        # Create a random index that respects the structure of an rdm.
        random.seed(i)
        sample = np.random.choice(eeg_rdm_vec.shape[0], eeg_rdm_vec.shape[0],replace=True) 

        # Subsample from both the reference and the feature RDM when calculating R2 for different models
        eeg_rdm_sample = eeg_rdm_vec[sample] 
        r2_scores = {}
        models = design_matrices.keys()
        for model in models:
            design_matrix = design_matrices[model]
            design_matrix_sample = design_matrix[sample]
            model_fit = LinearRegression().fit(design_matrix_sample, eeg_rdm_sample)
            r2 = model_fit.score(design_matrix_sample, eeg_rdm_sample)
            r2_scores[model] = r2*100

        # Perform variance partitioning 
        u_a_r2 = r2_scores['o-s-a'] - r2_scores['o-s']
        u_s_r2 = r2_scores['o-s-a'] - r2_scores['o-a']
        u_o_r2 = r2_scores['o-s-a'] - r2_scores['s-a']

        os_shared_r2 = r2_scores['o-s-a'] - r2_scores['a'] - u_o_r2 - u_s_r2
        sa_shared_r2 = r2_scores['o-s-a'] - r2_scores['o'] - u_s_r2 - u_a_r2
        oa_shared_r2 = r2_scores['o-s-a'] - r2_scores['s'] - u_o_r2 - u_a_r2

        osa_shared_r2 = r2_scores['o-s-a'] - u_o_r2 - u_s_r2 - u_a_r2 - os_shared_r2 - sa_shared_r2 - oa_shared_r2

        o_total_r2 = u_o_r2 +  os_shared_r2 + oa_shared_r2 + osa_shared_r2
        s_total_r2 = u_s_r2 + os_shared_r2 + sa_shared_r2 + osa_shared_r2
        a_total_r2 = u_a_r2 + oa_shared_r2 + sa_shared_r2 + osa_shared_r2

        all_parts = u_o_r2 + u_s_r2 + u_a_r2 + os_shared_r2 + sa_shared_r2 + oa_shared_r2 + osa_shared_r2

        vp_boot_scores['u_a'].append(u_a_r2)
        vp_boot_scores['u_s'].append(u_s_r2)
        vp_boot_scores['u_o'].append(u_o_r2)
        vp_boot_scores['os_shared'].append(os_shared_r2)
        vp_boot_scores['sa_shared'].append(sa_shared_r2)
        vp_boot_scores['oa_shared'].append(oa_shared_r2)
        vp_boot_scores['osa_shared'].append(osa_shared_r2)
        vp_boot_scores['o_total'].append(o_total_r2)
        vp_boot_scores['s_total'].append(s_total_r2)
        vp_boot_scores['a_total'].append(a_total_r2)
        vp_boot_scores['all_parts'].append(all_parts)

    # Get 95 % confidence intervals
    ci_vals= {}
    ci_vals['u_a'] = (np.percentile(vp_boot_scores['u_a'], 2.5), np.percentile(vp_boot_scores['u_a'], 97.5))
    ci_vals['u_o'] = (np.percentile(vp_boot_scores['u_o'], 2.5), np.percentile(vp_boot_scores['u_o'], 97.5))
    ci_vals['u_s'] = (np.percentile(vp_boot_scores['u_s'], 2.5), np.percentile(vp_boot_scores['u_s'], 97.5))
    ci_vals['os_shared'] = (np.percentile(vp_boot_scores['os_shared'], 2.5), np.percentile(vp_boot_scores['os_shared'], 97.5))
    ci_vals['sa_shared'] = (np.percentile(vp_boot_scores['sa_shared'], 2.5), np.percentile(vp_boot_scores['sa_shared'], 97.5))
    ci_vals['oa_shared'] = (np.percentile(vp_boot_scores['oa_shared'], 2.5), np.percentile(vp_boot_scores['oa_shared'], 97.5))
    ci_vals['osa_shared'] = (np.percentile(vp_boot_scores['oa_shared'], 2.5), np.percentile(vp_boot_scores['osa_shared'], 97.5))
    ci_vals['a_total'] = (np.percentile(vp_boot_scores['a_total'], 2.5), np.percentile(vp_boot_scores['a_total'], 97.5))
    ci_vals['o_total'] = (np.percentile(vp_boot_scores['o_total'], 2.5), np.percentile(vp_boot_scores['o_total'], 97.5))
    ci_vals['s_total'] = (np.percentile(vp_boot_scores['s_total'], 2.5), np.percentile(vp_boot_scores['s_total'], 97.5))
    ci_vals['all_parts'] = (np.percentile(vp_boot_scores['all_parts'], 2.5), np.percentile(vp_boot_scores['all_parts'], 97.5))


    toc = time.time()
    print(f'iteration {t} in {toc-tic}')

    return ci_vals



def calc_rsquared_rw(t, data_shape, dtype, design_matrix, shm, ridge, cv):
    """ Calculates variance explained between eeg rdm and model rdm at time point t. 
    
    ---> Outdated

    Parameters
    ----------
    t: int
        Time point of interest
    data_shape: shape
        Data shape of EEG data
    dtype: type
        Data type of EEG data
    design_matrix: array of floats
       Combination of flattened model rdms of interest
    shm: str
        Name of shared memory.
    ridge: int
        To perform ridge regression y/n
    cv: int
        To perform cross validation y/n

    Returns
    -------
    (r2, p-value): (float, float)
        Variance explained, p-value.
    """

    print(f'Calculating r^2 t= {t}')

    existing_shm = shared_memory.SharedMemory(name=shm)
    eeg_rdms = np.ndarray(data_shape, dtype=dtype, buffer=existing_shm.buf)

    tic = time.time()

    eeg_rdm = eeg_rdms[:, :, t]
    sq_eeg_rdm  = squareform(eeg_rdm, checks=False)

    # Reweighting of model rdms using multiple linear regression
    if ridge == 0 and cv == 0:
        model = LinearRegression().fit(design_matrix, sq_eeg_rdm)
        r2 = model.score(design_matrix, sq_eeg_rdm)
    if ridge == 0 and cv == 1:
        kf = KFold(n_splits=sq_eeg_rdm.shape[0])
        r2_scores = []
        for train, test in kf.split(sq_eeg_rdm):
            X_train, X_test, Y_train, Y_test = design_matrix[train], design_matrix[test], sq_eeg_rdm[train], sq_eeg_rdm[test]
            model = LinearRegression().fit(X_train, Y_train)
            predictions = model.predict(X_test)
            r2 = r2_score(Y_test, predictions)
            r2_scores.append(r2)
        r2 = np.mean(r2_scores)
    if ridge == 1 and cv == 0:
        model = Ridge().fit(design_matrix, sq_eeg_rdm)
        r2 = model.score(design_matrix, sq_eeg_rdm)
    if ridge == 1 and cv == 1:
        kf = KFold(n_splits=sq_eeg_rdm.shape[0])
        r2_scores = []
        for train, test in kf.split(sq_eeg_rdm):
            X_train, X_test, Y_train, Y_test = design_matrix[train], design_matrix[test], sq_eeg_rdm[train], sq_eeg_rdm[test]
            model = LinearRegression().fit(X_train, Y_train)
            predictions = model.predict(X_test)
            r2 = r2_score(Y_test, predictions)
            r2_scores.append(r2)
        r2 = np.mean(r2_scores)
    
    rdm_cor_p = np.nan

    toc = time.time()
    print(f't= {t} done in {toc-tic}')

    return (r2, rdm_cor_p)


def calc_rsquared_rw_2(t, data_shape, dtype, design_matrices, its, shm):
    """ Peforms variance partitioning between eeg rdm and models at time point t. 
    
    Parameters
    ----------
    t: int
        Time point of interest
    data_shape: shape
        Data shape of EEG data
    dtype: type
        Data type of EEG data
    design_matrices: array of floats
       Combination of flattened model rdms of interest
    its: int
        n of iterations used to compute significance
    shm: str
        Name of shared memory.

    Returns
    -------
    (lower_p, upper_p): (float, float)
        Lower percentile, upper percentile
    """


    print(f'Calculating r^2 t= {t}')
    tic = time.time()

    existing_shm = shared_memory.SharedMemory(name=shm)
    eeg_rdms = np.ndarray(data_shape, dtype=dtype, buffer=existing_shm.buf)
    eeg_rdm = eeg_rdms[:, :, t]
    sq_eeg_rdm  = squareform(eeg_rdm, checks=False)

    # Calculate scores for all models
    r2_scores = {}
    models = design_matrices.keys()
    for model in models:
        design_matrix = design_matrices[model]

        model_fit = LinearRegression().fit(design_matrix, sq_eeg_rdm)
        r2 = model_fit.score(design_matrix, sq_eeg_rdm)
        r2_scores[model] = r2*100
    
    # Perform variance partitioning 
    u_a_r2 = r2_scores['o-s-a'] - r2_scores['o-s']
    u_s_r2 = r2_scores['o-s-a'] - r2_scores['o-a']
    u_o_r2 = r2_scores['o-s-a'] - r2_scores['s-a']

    os_shared_r2 = r2_scores['o-s-a'] - r2_scores['a'] - u_o_r2 - u_s_r2
    sa_shared_r2 = r2_scores['o-s-a'] - r2_scores['o'] - u_s_r2 - u_a_r2
    oa_shared_r2 = r2_scores['o-s-a'] - r2_scores['s'] - u_o_r2 - u_a_r2
    osa_shared_r2 = r2_scores['o-s-a'] - u_o_r2 - u_s_r2 - u_a_r2 - os_shared_r2 - sa_shared_r2 - oa_shared_r2

    o_total_r2 = u_o_r2 +  os_shared_r2 + oa_shared_r2 + osa_shared_r2
    s_total_r2 = u_s_r2 + os_shared_r2 + sa_shared_r2 + osa_shared_r2
    a_total_r2 = u_a_r2 + oa_shared_r2 + sa_shared_r2 + osa_shared_r2
    all_parts = u_o_r2 + u_s_r2 + u_a_r2 + os_shared_r2 + sa_shared_r2 + oa_shared_r2 + osa_shared_r2

    vp_scores = {
        'u_a': u_a_r2,
        'u_s': u_s_r2,
        'u_o': u_o_r2,
        'os_shared' : os_shared_r2,
        'sa_shared' : sa_shared_r2,
        'oa_shared' : oa_shared_r2,
        'osa_shared' : osa_shared_r2,
        'o_total': o_total_r2,
        's_total': s_total_r2,
        'a_total' : a_total_r2,
        'all_parts' : all_parts
    }

    p_vals = corr_nullDist_rw(vp_scores, design_matrices, eeg_rdm, its=its)

    return (vp_scores, p_vals)





