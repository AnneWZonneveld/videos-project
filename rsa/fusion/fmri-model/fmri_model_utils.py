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



def corr_nullDist(rdm_cor, rdm_1, rdm_2, its=100, eval_method='spearman'):
    """ Calculates null distribution based on permutations for correlation calculation. 
    Sub level.

    Parameters
    ----------
    rdm_cor: float 
        Found experimental correlation value
    rdm_1 : float array
        RDM 1 (= fMRI RDM) (conditions, conditions)
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

        # Create a random index that respects the structure of an rdm.
        random.seed(i)
        shuffle = np.random.choice(rdv_1.shape[0], rdv_1.shape[0],replace=False)

        # shuffle RDM consistently for both dims
        shuffled_rdm_1 = rdv_1[shuffle]

        # correlating with neural similarty matrix
        if eval_method == 'spearman':
            rdm_corr_null.append(spearmanr(shuffled_rdm_1, squareform(rdm_2, checks=False))[0])
        elif eval_method == 'pearson':
            rdm_corr_null.append(pearsonr(shuffled_rdm_1, squareform(rdm_2, checks=False))[0])

    # Calc p value (two sided)
    rdm_corr_null = np.array(rdm_corr_null)
    p_val = np.mean(abs(rdm_corr_null)>abs(rdm_cor))
    print(f'p_value: {p_val}')

    return p_val

def corr_nullDist_GA(rdm_cor, rdm_1, rdm_2, noise_ceiling, its=100, eval_method='spearman'):
    """ Calculates null distribution based on permutations for correlation calculation. 
    Group level --> normalization using noise ceiling. 

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

        # Create a random index that respects the structure of an rdm.
        random.seed(i)
        shuffle = np.random.choice(rdv_1.shape[0], rdv_1.shape[0],replace=False)

        # shuffle RDM consistently for both dims
        shuffled_rdm_1 = rdv_1[shuffle]

        # correlating with neural similarty matrix
        if eval_method == 'spearman':
            rdm_corr_null.append(spearmanr(shuffled_rdm_1, squareform(rdm_2, checks=False))[0] / noise_ceiling['UpperBound'])
        elif eval_method == 'pearson':
            rdm_corr_null.append(pearsonr(shuffled_rdm_1, squareform(rdm_2, checks=False))[0] / noise_ceiling['UpperBound'])

    # Calc p value (two sided)
    rdm_corr_null = np.array(rdm_corr_null)
    p_val = np.mean(abs(rdm_corr_null)>abs(rdm_cor))
    print(f'p_value: {p_val}')

    return p_val


def corr_nullDist_boot(fmri_rdm, feature_rdm, its, eval_method='spearman'):
    """ Calculates null distribution based on bootstrapped distribution for 
    correlation calculation. To compare permutation-based p-value vs bootstrapped based p-value.
    Sub level.

    Parameters
    ----------
    rdm_cor: float 
        Found experimental correlation value
    rdm_1 : float array
        RDM 1 (= fMRI RDM) (conditions, conditions)
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

    tic = time.time()

    fmri_rdm_vec = squareform(fmri_rdm, checks=False)
    feature_rdm_vec = squareform(feature_rdm, checks=False)

    tic = time.time()

    rdm_corr_boots = []

    for i in range(its):

        if i % 100 == 0:
            print(f'cor var iteration: {i}')

        # Create a random index that respects the structure of an rdm.
        random.seed(i)
        sample = np.random.choice(fmri_rdm_vec.shape[0], fmri_rdm_vec.shape[0],replace=True) 

        # Subsample from both the reference and the feature RDM
        fmri_rdm_sample = fmri_rdm_vec[sample] 
        feature_rdm_sample = feature_rdm_vec[sample] 

        # correlating with neural similarty matrix
        if eval_method == 'spearman':
            rdm_corr_boots.append(spearmanr(fmri_rdm_sample, feature_rdm_sample)[0])
        elif eval_method == 'pearson':
            rdm_corr_boots.append(pearsonr(fmri_rdm_sample, feature_rdm_sample)[0])
        elif eval_method == 'euclidean':
            rdm_corr_boots.append(np.linalg.norm(fmri_rdm_sample - feature_rdm_sample))

    # Calc p value (two sided)
    rdm_corr_null = np.array(rdm_corr_boots)
    p_val = np.mean(abs(rdm_corr_null)>abs(0))

    toc = time.time()

    return p_val


def calc_corr(roi, fmri_data, feature_rdm, its, eval_method='spearman'):
    """ Calculates correlation calculation fMRI RDM and model RDM. 
    Sub level.

    Parameters
    ----------
    roi: str
        ROI name. 
    fmri_data : dict of arrays
        Dictionary with RDMS for all ROIs.
    feature_rdm : float array
        Model RDM (conditions, conditions)
    its: int
        n of iterations used to compute significance
    eval_method: str
        Evaluation method to calculate similarity; 'spearman'/'pearson'

    Returns
    -------
    (rdm_cor, p-val): (float, float)
        Correlation value, P-value.
    """

    print(f'Calculating cor {roi}')
    tic = time.time()

    try:
        fmri_rdm=fmri_data[roi]

        if eval_method == 'spearman':
            rdm_cor = spearmanr(squareform(fmri_rdm, checks=False), squareform(feature_rdm, checks=False))[0] 
        elif eval_method == 'pearson':
            rdm_cor = pearsonr(squareform(fmri_rdm, checks=False), squareform(feature_rdm, checks=False))[0] 
        elif eval_method == 'euclidean': 
            rdm_cor = np.linalg.norm(squareform(fmri_rdm, checks=False) - squareform(feature_rdm, checks=False))
        
        rdm_cor_p = corr_nullDist(rdm_cor, fmri_rdm, feature_rdm, its=its, eval_method=eval_method) 

        toc = time.time()

        return (rdm_cor, rdm_cor_p)
    except:
        print(f'{roi} not present')

        return (np.nan, np.nan)


def calc_corr_GA(roi, fmri_data, feature_rdm, its, noise_ceilings, eval_method='spearman'):
    """ Calculates correlation calculation fMRI RDM and model RDM. 
    Group level ---> normalization using noise ceiling. 

    Parameters
    ----------
    roi: str
        ROI name. 
    fmri_data : dict of arrays
        Dictionary with RDMS for all ROIs.
    feature_rdm : float array
        Model RDM (conditions, conditions)
    its: int
        n of iterations used to compute significance
    eval_method: str
        Evaluation method to calculate similarity; 'spearman'/'pearson'

    Returns
    -------
    (rdm_cor, p-val): (float, float)
        Correlation value, P-value.
    """

    print(f'Calculating cor {roi}')
    tic = time.time()

    try:
        fmri_rdm=fmri_data[roi]
        noise_ceiling = noise_ceilings[roi]

        if eval_method == 'spearman':
            rdm_cor = spearmanr(squareform(fmri_rdm, checks=False), squareform(feature_rdm, checks=False))[0] 
        elif eval_method == 'pearson':
            rdm_cor = pearsonr(squareform(fmri_rdm, checks=False), squareform(feature_rdm, checks=False))[0] 
        elif eval_method == 'euclidean': 
            rdm_cor = np.linalg.norm(squareform(fmri_rdm, checks=False) - squareform(feature_rdm, checks=False))
        
        rdm_cor = rdm_cor / noise_ceiling['UpperBound']
        rdm_cor_p = corr_nullDist_GA(rdm_cor, fmri_rdm, feature_rdm, noise_ceiling, its=its, eval_method=eval_method) 

        toc = time.time()

        return (rdm_cor, rdm_cor_p)
    except:
        print(f'{roi} not present')

        return (np.nan, np.nan)

# def calc_corr_GA_boot(roi, fmri_data, feature_rdm, its, noise_ceilings, eval_method='spearman'):

#     print(f'Calculating cor {roi}')
#     tic = time.time()

#     try:
#         fmri_rdm=fmri_data[roi]
#         noise_ceiling = noise_ceilings[roi]

#         if eval_method == 'spearman':
#             rdm_cor = spearmanr(squareform(fmri_rdm, checks=False), squareform(feature_rdm, checks=False))[0] 
#         elif eval_method == 'pearson':
#             rdm_cor = pearsonr(squareform(fmri_rdm, checks=False), squareform(feature_rdm, checks=False))[0] 
#         elif eval_method == 'euclidean': 
#             rdm_cor = np.linalg.norm(squareform(fmri_rdm, checks=False) - squareform(feature_rdm, checks=False))
        
#         rdm_cor = rdm_cor / noise_ceiling['UpperBound']
#         rdm_cor_p = corr_nullDist_boot(fmri_rdm=fmri_rdm, feature_rdm=feature_rdm, its=its, eval_method=eval_method) 

#         toc = time.time()

#         return (rdm_cor, rdm_cor_p)
#     except:
#         print(f'{roi} not present')

#         return (np.nan, np.nan)


def calc_corr_boot(roi, fmri_data, feature_rdm, its, eval_method='spearman'):
    """ Calculates correlation calculation fMRI RDM and model RDM. P value determined using bootstrapped-based 
    distribution.
    Sub level.

    Parameters
    ----------
    roi: str
        ROI name. 
    fmri_data : dict of arrays
        Dictionary with RDMS for all ROIs.
    feature_rdm : float array
        Model RDM (conditions, conditions)
    its: int
        n of iterations used to compute significance
    eval_method: str
        Evaluation method to calculate similarity; 'spearman'/'pearson'

    Returns
    -------
    (rdm_cor, p-val): (float, float)
        Correlation value, P-value.
    """

    print(f'Calculating cor {roi}')
    tic = time.time()

    try:
        fmri_rdm=fmri_data[roi]

        if eval_method == 'spearman':
            rdm_cor = spearmanr(squareform(fmri_rdm, checks=False), squareform(feature_rdm, checks=False))[0] 
        elif eval_method == 'pearson':
            rdm_cor = pearsonr(squareform(fmri_rdm, checks=False), squareform(feature_rdm, checks=False))[0] 
        elif eval_method == 'euclidean': 
            rdm_cor = np.linalg.norm(squareform(fmri_rdm, checks=False) - squareform(feature_rdm, checks=False))
        
        rdm_cor_p = corr_nullDist_boot(fmri_rdm=fmri_rdm, feature_rdm=feature_rdm, its=its, eval_method=eval_method) 

        toc = time.time()

        return (rdm_cor, rdm_cor_p)
    
    except:
        print(f'{roi} not present')

        return (np.nan, np.nan)


def cor_variability(roi, fmri_data, feature_rdm, its, eval_method='spearman'):
    """ Calculates 95% confidence intervaal between fMRI RDM and model RDM. 
    Sub level.

    Parameters
    ----------
    roi: str
        ROI name. 
    fmri_data : dict of arrays
        Dictionary with RDMS for all ROIs.
    feature_rdm : float array
        Model RDM (conditions, conditions)
    its: int
        n of iterations used to compute significance
    eval_method: str
        Evaluation method to calculate similarity; 'spearman'/'pearson'

    Returns
    -------
    (lower_p, upper_p): (float, float)
        Lower percentile, upper percentile.
    """

    print(f'Calculating cis {roi}')
    tic = time.time()

    try: 
        fmri_rdm=fmri_data[roi]

        fmri_rdm_vec = squareform(fmri_rdm, checks=False)
        feature_rdm_vec = squareform(feature_rdm, checks=False)

        tic = time.time()

        rdm_corr_boots = []

        for i in range(its):

            if i % 100 == 0:
                print(f'cor var iteration: {i}')

            # Create a random index that respects the structure of an rdm.
            random.seed(i)
            sample = np.random.choice(fmri_rdm_vec.shape[0], fmri_rdm_vec.shape[0],replace=True) 

            # Subsample from both the reference and the feature RDM
            fmri_rdm_sample = fmri_rdm_vec[sample] 
            feature_rdm_sample = feature_rdm_vec[sample] 

            # correlating with neural similarty matrix
            if eval_method == 'spearman':
                rdm_corr_boots.append(spearmanr(fmri_rdm_sample, feature_rdm_sample)[0])
            elif eval_method == 'pearson':
                rdm_corr_boots.append(pearsonr(fmri_rdm_sample, feature_rdm_sample)[0])
            elif eval_method == 'euclidean':
                rdm_corr_boots.append(np.linalg.norm(fmri_rdm_sample - feature_rdm_sample))


        # Get 95% confidence interval
        lower_p = np.percentile(rdm_corr_boots, 2.5)
        upper_p = np.percentile(rdm_corr_boots, 97.5)

        toc = time.time()

        return (lower_p, upper_p)
    except:
        print(f'{roi} not present')

        return (np.nan, np.nan)
    
def cor_variability_GA(roi, fmri_data, feature_rdm, its, noise_ceilings, eval_method='spearman'):
    """ Calculates 95% confidence interval between fMRI RDM and model RDM. 
    Group level ---> normalization using noise ceiling.

    Parameters
    ----------
    roi: str
        ROI name. 
    fmri_data : dict of arrays
        Dictionary with RDMS for all ROIs.
    feature_rdm : float array
        Model RDM (conditions, conditions)
    its: int
        n of iterations used to compute significance
    eval_method: str
        Evaluation method to calculate similarity; 'spearman'/'pearson'

    Returns
    -------
    (lower_p, upper_p): (float, float)
        Lower percentile, upper percentile.
    """

    print(f'Calculating cis {roi}')
    tic = time.time()

    try: 
        fmri_rdm=fmri_data[roi]
        noise_ceiling = noise_ceilings[roi]

        fmri_rdm_vec = squareform(fmri_rdm, checks=False)
        feature_rdm_vec = squareform(feature_rdm, checks=False)

        tic = time.time()

        rdm_corr_boots = []

        for i in range(its):

            if i % 100 == 0:
                print(f'cor var iteration: {i}')

            # Create a random index that respects the structure of an rdm.
            random.seed(i)
            sample = np.random.choice(fmri_rdm_vec.shape[0], fmri_rdm_vec.shape[0],replace=True) 

            # Subsample from both the reference and the feature RDM
            fmri_rdm_sample = fmri_rdm_vec[sample] 
            feature_rdm_sample = feature_rdm_vec[sample] 

            # correlating with neural similarty matrix
            if eval_method == 'spearman':
                rdm_corr = spearmanr(fmri_rdm_sample, feature_rdm_sample)[0]
            elif eval_method == 'pearson':
                rdm_corr = pearsonr(fmri_rdm_sample, feature_rdm_sample)[0]
            
            rdm_corr = rdm_corr / noise_ceiling['UpperBound']
            rdm_corr_boots.append(rdm_corr)


        # Get 95% confidence interval
        lower_p = np.percentile(rdm_corr_boots, 2.5)
        upper_p = np.percentile(rdm_corr_boots, 97.5)

        toc = time.time()

        return (lower_p, upper_p)
    except:
        print(f'{roi} not present')

        return (np.nan, np.nan)


def corr_nullDist_rw(vp_scores, design_matrices, fmri_rdm, its=100, inspect=False):
    """ Calculates null distribution based on permutations for variance partitioning. 

    Parameters
    ----------
    vp_scores: dict of floats
        Found experimental variance explained values
    design_matrices: array of floats
        Combination of flattened model RDMs.
    fmri_rdm : float array
        fMRI RDM (conditions, conditions)
    its: int
        n of iterations used to compute significance
    inspect: bool
        To make inspection plots y/n

    Returns
    -------
    p-vals: dict of floats
        Dictionary of p-values for all partitions.
    """

    print("Constructing null distribution")
    
    fmri_rdv = squareform(fmri_rdm, checks=False)

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
        shuffle = np.random.choice(fmri_rdv.shape[0], fmri_rdv.shape[0], replace=False)

        shuffled_fmri = fmri_rdv[shuffle] 
    
        # Calc r2 for different models
        r2_scores = {}
        models = design_matrices.keys()
        for model in models:
            design_matrix = design_matrices[model]
            model_fit = LinearRegression().fit(design_matrix, shuffled_fmri)
            r2 = model_fit.score(design_matrix, shuffled_fmri)
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

    if inspect == True:

        keys = p_vals.keys()
        for key in keys:

            fig = sns.displot(vp_perm_scores[key])
            fig.set_axis_labels("r2")
            plt.axvline(vp_scores[key], color="red")

            fig.tight_layout()

            res_folder  = f'/scratch/azonneveld/rsa/fusion/fmri-model/plots/vp-results/train/model_euclidean/GA/pearson/'
            if not os.path.exists(res_folder) == True:
                os.makedirs(res_folder)

            img_path = res_folder + f'null_{its}_LOC_{key}.png'
            plt.savefig(img_path)
            plt.clf()

    
    return p_vals


def calc_rsquared_rw(roi, fmri_data, design_matrices, its=10, inspect=False):
    """ Calculates variance explained values and p-values
    based on permutations for variance partitioning. 

    Parameters
    ----------
    roi: str
        Name of ROI
    fmri_data: dict of arrays
        Dictionary with RDMs for all ROIs.
    design_matrices: array of floats
        Combination of flattened model RDMs.
    its: int
        n of iterations used to compute significance
    inspect: bool
        To make inspection plots y/n

    Returns
    -------
    (vp_scores, p_vals): (dict, dict)
        Dictionary of variance explained scores, dictionary of p-values for all partitions.
    """

    print(f'Calculating r^2 {roi}')
    tic = time.time()

    fmri_rdm = fmri_data[roi]
    sq_fmri_rdm  = squareform(fmri_rdm, checks=False)

    # Calculate scores for all models
    r2_scores = {}
    models = design_matrices.keys()
    for model in models:
        design_matrix = design_matrices[model]

        model_fit = LinearRegression().fit(design_matrix, sq_fmri_rdm)
        r2 = model_fit.score(design_matrix, sq_fmri_rdm)
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

    p_vals = corr_nullDist_rw(vp_scores=vp_scores, design_matrices=design_matrices, fmri_rdm=fmri_rdm, its=its, inspect=inspect)

    return (vp_scores, p_vals)


def calc_variability_rw(roi, fmri_data, design_matrices, its=100):
    """ Calculates 95% confidence intervals fo variance explained values 
    for variance partitioning.

    --> unreliable results

    Parameters
    ----------
    roi: str
        Name of ROI
    fmri_data: dict of arrays
        Dictionary with RDMs for all ROIs.
    design_matrices: array of floats
        Combination of flattened model RDMs.
    its: int
        n of iterations used to compute significance

    Returns
    -------
    ci_vals: dict of tuples
        Dictionary for all variance partitions with tuples (lower percentile, upper percentile).
    """

    print(f'Calculating r^2 {roi}')
    tic = time.time()

    fmri_rdm = fmri_data[roi]
    sq_fmri_rdm  = squareform(fmri_rdm, checks=False)
    
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
        sample = np.random.choice(sq_fmri_rdm.shape[0], sq_fmri_rdm.shape[0],replace=True) 

        # Subsample from both the reference and the feature RDM when calculating R2 for different models
        fmri_rdm_sample = sq_fmri_rdm[sample] 
        r2_scores = {}
        models = design_matrices.keys()
        for model in models:
            design_matrix = design_matrices[model]
            design_matrix_sample = design_matrix[sample]
            model_fit = LinearRegression().fit(design_matrix_sample, fmri_rdm_sample)
            r2 = model_fit.score(design_matrix_sample, fmri_rdm_sample)
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
    ci_vals['osa_shared'] = (np.percentile(vp_boot_scores['osa_shared'], 2.5), np.percentile(vp_boot_scores['osa_shared'], 97.5))
    ci_vals['a_total'] = (np.percentile(vp_boot_scores['a_total'], 2.5), np.percentile(vp_boot_scores['a_total'], 97.5))
    ci_vals['o_total'] = (np.percentile(vp_boot_scores['o_total'], 2.5), np.percentile(vp_boot_scores['o_total'], 97.5))
    ci_vals['s_total'] = (np.percentile(vp_boot_scores['s_total'], 2.5), np.percentile(vp_boot_scores['s_total'], 97.5))
    ci_vals['all_parts'] = (np.percentile(vp_boot_scores['all_parts'], 2.5), np.percentile(vp_boot_scores['all_parts'], 97.5))


    toc = time.time()
    print(f'iteration {roi} in {toc-tic}')

    return ci_vals

