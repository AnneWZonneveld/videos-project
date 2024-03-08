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
    """
        Calculates null distribution based on permutations. 
    """
    print("Constructing null distribution")

    rdm_corr_null = []
    for i in range(its):

        if i % 1000 == 0:
            print(f'null distr iteration: {i}')

        # Create a random index that respects the structure of an rdm.
        random.seed(i)
        shuffle = np.random.choice(rdm_1.shape[0], rdm_1.shape[0],replace=False)

        # shuffle RDM consistently for both dims
        shuffled_rdm_1 = rdm_1[shuffle,:] # rows
        shuffled_rdm_1 = shuffled_rdm_1[:,shuffle] # columns

        # correlating with neural similarty matrix
        if eval_method == 'spearman':
            rdm_corr_null.append(spearmanr(squareform(shuffled_rdm_1, checks=False), squareform(rdm_2, checks=False))[0])
        elif eval_method == 'pearson':
            rdm_corr_null.append(pearsonr(squareform(shuffled_rdm_1, checks=False), squareform(rdm_2, checks=False))[0])
        elif eval_method == 'euclidean':
            rdm_cor = np.linalg.norm(squareform(shuffled_rdm_1, checks=False) - squareform(rdm_2, checks=False))
    
    # Calc p value (two sided)
    rdm_corr_null = np.array(rdm_corr_null)
    p_val = np.mean(abs(rdm_corr_null)>abs(rdm_cor), dtype='float64')
    print(f'p_value: {p_val}')

    # # Plot null distribution
    # cor = spearmanr(squareform(rdm_1, checks=False), squareform(rdm_2, checks=False))[0]

    # fig, axes = plt.subplots(dpi=300)
    # sns.displot(rdm_corr_null, bins=50)
    # axes.set_title(f'Null distr')
    # axes.set_ylabel('Spearman')

    # fig = sns.displot(rdm_corr_null, bins=50)
    # fig.set_axis_labels("Spearman")
    # plt.axvline(cor, color="red")

    # fig.tight_layout()

    # res_folder  = f'/scratch/azonneveld/rsa/fusion/fmri-model/plots/standard/test/model_euclidean/GA/pearson/spearman/'
    # if not os.path.exists(res_folder) == True:
    #     os.makedirs(res_folder)

    # img_path = res_folder + f'null_{its}_V1v_actions.png'
    # plt.savefig(img_path)
    # plt.clf()


    return p_val


def corr_nullDist_boot(fmri_rdm, feature_rdm, its, eval_method='spearman'):

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


def calc_corr_boot(roi, fmri_data, feature_rdm, its, eval_method='spearman'):

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

        # # Temp
        # fig, axes = plt.subplots(dpi=300)
        # sns.displot(rdm_corr_boots, bins=50)
        # axes.set_title(f'Bootstrap dist sub {sub}, t={t}')
        # axes.set_ylabel('Spearman')

        # fig.tight_layout()

        # img_path = f'/scratch/azonneveld/rsa/fusion/eeg-model/standard/plots/z_0/sub-01/cor-dist-{t}'
        # plt.savefig(img_path)
        # plt.clf()

        toc = time.time()

        return (lower_p, upper_p)
    except:
        print(f'{roi} not present')

        return (np.nan, np.nan)


def corr_nullDist_rw(vp_scores, design_matrices, fmri_rdm, its=100):
    """
        Calculates null distribution based on permutations. 
    """
    print("Constructing null distribution")
    
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
        shuffle = np.random.choice(fmri_rdm.shape[0], fmri_rdm.shape[0],replace=False)

        shuffled_fmri = fmri_rdm[shuffle, :] #rows
        shuffled_fmri = shuffled_fmri[:, shuffle] #columns
        sq_fmri_rdm = squareform(shuffled_fmri, checks=False)
        
        # Calc r2 for different models
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


def calc_rsquared_rw(roi, fmri_data, design_matrices, its=10):

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
        r2 = model_fit.score(design_matrix, fmri_rdm)
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

    p_vals = corr_nullDist_rw(vp_scores=vp_scores, design_matrices=design_matrices, fmri_rdm=fmri_rdm, its=its)

    return (vp_scores, p_vals)
