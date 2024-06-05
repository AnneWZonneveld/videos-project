import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from sklearn.utils import resample
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.svm import SVC
import argparse
from functools import partial
from multiprocessing import shared_memory
from multiprocessing import current_process, cpu_count
import time


def compute_rdm(t, data_shape, pseudo_order, shm, dtype, data_split='train', distance_type='euclidean', resampled=False):
    """ Compute EEG-based RDM for timepoint t.

	Parameters
	----------
	t : int
		Time point
	pseudo_order : int array
		Array with pseudo order of trials (timepoints, conditions)
	ts : int
		Number of timepoints
	shm : str
        Shared memory name
    dtype: type
        Data type of shared memory.
	data_split: str
        Train or test. 
    distance_type: str
        Distance type: euclidean, pearson, euclidean-cv, classification or dv-classification
    resampled: bool
        Concerning resampled test RDM y/n.  

	Returns
	-------
	rdm_array: float array
        RDM at timepoint t.

	"""

    print(f'compute rdm for t={t}')
    tic = time.time()

    if data_split=='train':
        n_conditions = 1000
    elif data_split=='test':
        n_conditions = 102

    existing_shm = shared_memory.SharedMemory(name=shm)
    eeg_data = np.ndarray(data_shape, dtype=dtype, buffer=existing_shm.buf)
    
    rdm_array = np.zeros((n_conditions, n_conditions))
    
    # Loop through triangle of RDM
    combination = 0 
    for v1 in range(n_conditions):
        for v2 in range(v1):
            idx_1 = pseudo_order[combination][0]
            idx_2 = pseudo_order[combination][1]
            eeg_cond_1 = eeg_data[idx_1,:,t]
            eeg_cond_2 = eeg_data[idx_2,:,t]

            # Select a minimum amount of trials in case repeats are not the 
            # same between two test image conditions
            if len(eeg_cond_1) < len(eeg_cond_2):
                eeg_cond_2 = eeg_cond_2[:len(eeg_cond_1)]
            elif len(eeg_cond_2) < len(eeg_cond_1):
                eeg_cond_1 = eeg_cond_1[:len(eeg_cond_2)]
            
            # Create pseudo-trials
            if resampled == True:
                sample = np.random.choice(eeg_cond_1.shape[0], 6, replace=True) 
                re_eeg_cond_1 = eeg_cond_1[sample, :]
                re_eeg_cond_2 = eeg_cond_2[sample, :]
                eeg_cond_1 = re_eeg_cond_1
                eeg_cond_2 = re_eeg_cond_2
                n_ptrials_repeats = 3
            else:
                if data_split == 'test':
                    n_ptrials_repeats = 12
                elif data_split == 'train':
                    n_ptrials_repeats = 3

            n_pseudo_trials = int(np.ceil(len(eeg_cond_1) / n_ptrials_repeats))
            pseudo_data_1 = np.zeros((n_pseudo_trials, eeg_cond_1.shape[1]))
            pseudo_data_2 = np.zeros((n_pseudo_trials, eeg_cond_2.shape[1]))

            # Average pseudotrials
            for r in range(n_pseudo_trials):
                idx_start = r * n_ptrials_repeats
                idx_end = idx_start + n_ptrials_repeats
                pseudo_data_1[r] = np.mean(eeg_cond_1[idx_start:idx_end],0)
                pseudo_data_2[r] = np.mean(eeg_cond_2[idx_start:idx_end],0)
            
            eeg_cond_1 = pseudo_data_1
            eeg_cond_2 = pseudo_data_2

            if distance_type == 'euclidean':
                eeg_cond_1 = np.mean(eeg_cond_1,0)
                eeg_cond_2 = np.mean(eeg_cond_2,0)
                distance = np.linalg.norm(eeg_cond_1-eeg_cond_2)
                 
                rdm_array[v1, v2] = distance
                rdm_array[v2, v1] = distance

                del eeg_cond_1, eeg_cond_2
            
            if distance_type == 'pearson':
                eeg_cond_1 = np.mean(eeg_cond_1,0)
                eeg_cond_2 = np.mean(eeg_cond_2,0)
                distance =  1 - pearsonr(eeg_cond_1, eeg_cond_2)[0]
                 
                rdm_array[v1, v2] = distance
                rdm_array[v2, v1] = distance

                del eeg_cond_1, eeg_cond_2
            
            if distance_type == 'pearson-cv':

                cv_distances = np.zeros(len(eeg_cond_1))

                for r in range(len(eeg_cond_1)):

                    # Define the training/test partitions (LOOCV)
                    train_cond_1 = np.delete(eeg_cond_1, r, 0)
                    train_cond_2 = np.delete(eeg_cond_2, r, 0)

                    if len(train_cond_1.shape) > 2 :
                        train_cond_1 = np.mean(train_cond_1, 0)
                        train_cond_2 = np.mean(train_cond_2, 0)
                    
                    train_cond_1 = train_cond_1.squeeze()
                    train_cond_2 = train_cond_2.squeeze()
                    test_cond_1 = eeg_cond_1[r]
                    test_cond_2 = eeg_cond_2[r]

                    A1 = train_cond_1
                    A2 = test_cond_1
                    B1 = train_cond_2
                    B2 = test_cond_2

                    var_A = np.var(eeg_cond_1)
                    var_B = np.var(eeg_cond_2)
                    denom_noncv = np.sqrt(var_A * var_B)

                    cov_A1B2 = np.cov(A1, B2)[0,1] 
                    cov_B1A2 = np.cov(B1, A2)[0,1]
                    cov_AB = (cov_A1B2 + cov_B1A2) / 2

                    var_A12 = np.cov(A1, A2)[0,1] 
                    var_B12 = np.cov(B1, B2)[0,1] 
                    
                    # regularize variance
                    reg_factor_var = 0.1
                    denom = np.sqrt(max(reg_factor_var * var_A, var_A12) * max(reg_factor_var * var_B, var_B12))

                    # regularize denom
                    reg_factor_denom = 0.25
                    denom = max(reg_factor_denom * denom_noncv, denom)

                    cor = cov_AB / denom

                    # bound results
                    reg_bounding = 1
                    cor = min(max(-reg_bounding, cor), reg_bounding)

                    distance = 1 - cor
                    cv_distances[r] = distance
                
                distance = np.mean(cv_distances) 
                if np.isnan(distance) == True:
                    print('Encountered nan value')
                if np.isinf(distance) == True:
                    print('Encountered inf value')

                rdm_array[v1, v2] = distance
                rdm_array[v2, v1] = distance

            
            if distance_type == 'euclidean-cv':
                cv_distances = np.zeros(len(eeg_cond_1))

                for r in range(len(eeg_cond_1)):

                    # Define the training/test partitions (LOOCV)
                    train_cond_1 = np.delete(eeg_cond_1, r, 0)
                    train_cond_2 = np.delete(eeg_cond_2, r, 0)

                    if len(train_cond_1.shape) < 2:
                        dist_train = np.expand_dims(train_cond_1 - train_cond_2, 0)
                    else:
                        dist_train = np.expand_dims(np.mean(train_cond_1, 0) - np.mean(train_cond_2, 0), 0)
                    
                    test_cond_1 = np.expand_dims(eeg_cond_1[r], 0)
                    test_cond_2 = np.expand_dims(eeg_cond_2[r], 0)
                    dist_test = test_cond_1  - test_cond_2

                    cv_dist = np.dot(dist_train, np.transpose(dist_test))[0][0] #formula for squared cv euclidean distance?
                    cv_distances[r] = cv_dist
                
                distance = np.mean(cv_distances) 

                rdm_array[v1, v2] = distance
                rdm_array[v2, v1] = distance

            
            if distance_type in ['classification', 'dv-classification']:
                y_train = np.zeros(((len(eeg_cond_1)-1)*2))
                y_train[int(len(y_train)/2):] = 1
                y_test = np.asarray((0, 1))
                scores = np.zeros(len(eeg_cond_1))

                for r in range(len(eeg_cond_1)):

                    # Define the training/test partitions
                    X_train = np.append(np.delete(eeg_cond_1, r, 0),
                        np.delete(eeg_cond_2, r, 0), 0)
                    X_test = np.append(np.expand_dims(eeg_cond_1[r], 0),
                        np.expand_dims(eeg_cond_2[r], 0), 0)
                    
                    # Train the classifier
                    dec_svm = SVC(kernel='linear')
                    dec_svm.fit(X_train, y_train)
                    y_pred = dec_svm.predict(X_test)

                    # Test the classifier
                    if distance_type == 'dv-classification':
                        # Weighting of decision values (distance to hyper plane / norm weights), as in gugguenmos et al., 2018
                        decision_values = abs(dec_svm.decision_function(X_test) / np.linalg.norm(dec_svm.coef_))
                        weighted_score = sum(decision_values *((y_pred == y_test) - 0.5))/ len(y_test)
                        scores[r] = weighted_score
                    else: 
                        scores[r]= sum((y_pred == y_test) - 0.5)/ len(y_test)
                        # scores[r] = sum(y_pred == y_test) / len(y_test)

                score = np.mean(scores)
                
                rdm_array[v1, v2] = score
                rdm_array[v2, v1] = score

            combination = combination + 1
    
    toc = time.time()
    print(f'rdm for t={t} done ')
    print('timepoint done in {:.4f} seconds'.format(toc-tic))

    return rdm_array




