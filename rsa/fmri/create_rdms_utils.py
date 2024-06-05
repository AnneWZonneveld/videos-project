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


def compute_rdm(roi, fmri_data, pseudo_order, shm_name, data_split='test', distance_type='pearson'):
    """ Compute fmri-based RDM for ROI.

	Parameters
	----------
	ROI : str
        ROI
	fmri_data: dictionary 
        fMRI data for all ROIs
	pseudo_order : list of tuples
		Array with pseudo order of trials (conditions)
	shm_name : str
        Shared memory name
	data_split: str
        Train or test. 
    distance_type: str
        Distance type: euclidean, pearson, euclidean-cv, classification or dv-classification


	Returns
	-------
	(roi, rdm_array)
        Name of ROI, RDM for that ROI

	"""

    print(f'compute rdm for {roi}')
    tic = time.time()

    if data_split=='train':
        n_conditions = 1000
    elif data_split=='test':
        n_conditions = 102

    roi_data = fmri_data[roi]
    
    rdm_array = np.zeros((n_conditions, n_conditions))
    
    combination = 0 
    for v1 in range(n_conditions):
        for v2 in range(v1):
            idx_1 = pseudo_order[combination][0]
            idx_2 = pseudo_order[combination][1]
            fmri_cond_1 = roi_data[v1, idx_1]
            fmri_cond_2 = roi_data[v2, idx_2]

            # Select a minimum amount of trials in case repeats are not the 
            # same between two test image conditions
            if fmri_cond_1.shape[0] < fmri_cond_2.shape[0]:
                fmri_cond_2 = fmri_cond_2[:fmri_cond_1.shape[0]]
            elif fmri_cond_2.shape[0] < fmri_cond_1.shape[0]:
                fmri_cond_1 = fmri_cond_1[:fmri_cond_2.shape[0]]
            
            # Create pseudo-trials
            if data_split == 'test':
                n_ptrials_repeats = 5

                n_pseudo_trials = int(np.ceil(fmri_cond_1.shape[0] / n_ptrials_repeats))
                pseudo_data_1 = np.zeros((n_pseudo_trials, fmri_cond_1.shape[1]))
                pseudo_data_2 = np.zeros((n_pseudo_trials, fmri_cond_2.shape[1]))

                for r in range(n_pseudo_trials):
                    idx_start = r * n_ptrials_repeats
                    idx_end = idx_start + n_ptrials_repeats
                    pseudo_data_1[r] = np.mean(fmri_cond_1[idx_start:idx_end],0)
                    pseudo_data_2[r] = np.mean(fmri_cond_2[idx_start:idx_end],0)
            
                fmri_cond_1 = pseudo_data_1
                fmri_cond_2 = pseudo_data_2

            if distance_type == 'euclidean':
                fmri_cond_1 = np.mean(fmri_cond_1,0)
                fmir_cond_2 = np.mean(fmri_cond_2,0)
                distance = np.linalg.norm(fmri_cond_1-fmri_cond_2)
                 
                rdm_array[v1, v2] = distance
                rdm_array[v2, v1] = distance

                del fmri_cond_1, fmri_cond_2
            
            if distance_type == 'pearson':
                fmri_cond_1 = np.mean(fmri_cond_1,0)
                fmri_cond_2 = np.mean(fmri_cond_2,0)
                distance =  1 - pearsonr(fmri_cond_1, fmri_cond_2)[0]
                 
                rdm_array[v1, v2] = distance
                rdm_array[v2, v1] = distance

                del fmri_cond_1, fmri_cond_2
            
            if distance_type == 'euclidean-cv':
                cv_distances = np.zeros(fmri_cond_1.shape[0])

                for r in range(fmri_cond_1.shape[0]):

                    # Define the training/test partitions (LOOCV)
                    train_cond_1 = np.delete(fmri_cond_1, r, 0)
                    train_cond_2 = np.delete(fmri_cond_2, r, 0)

                    if train_cond_1.shape[0] < 2:
                        dist_train = np.expand_dims(train_cond_1 - train_cond_2, 0)
                    else:
                        dist_train = np.expand_dims(np.mean(train_cond_1, 0) - np.mean(train_cond_2, 0), 0)
                    
                    test_cond_1 = np.expand_dims(fmri_cond_1[r], 0)
                    test_cond_2 = np.expand_dims(fmri_cond_2[r], 0)
                    dist_test = test_cond_1  - test_cond_2

                    cv_dist = np.dot(dist_train, np.transpose(dist_test))[0][0] #formula for squared cv euclidean distance?
                    cv_distances[r] = cv_dist
                
                distance = np.mean(cv_distances) 

                rdm_array[v1, v2] = distance
                rdm_array[v2, v1] = distance

            if distance_type == 'mahalanobis-cv':

                cv_distances = np.zeros(fmri_cond_1.shape[0])

                for r in range(fmri_cond_1.shape[0]):

                    # Define the training/test partitions (LOOCV)
                    train_cond_1 = np.delete(fmri_cond_1, r, 0)
                    train_cond_2 = np.delete(fmri_cond_2, r, 0)

                    if train_cond_1.shape[0] < 2:
                        dist_train = np.expand_dims(train_cond_1 - train_cond_2, 0)
                    else:
                        dist_train = np.expand_dims(np.mean(train_cond_1, 0) - np.mean(train_cond_2, 0), 0)
                    
                    cov_train = np.cov(np.vstack((train_cond_1, train_cond_2)))
                    
                    test_cond_1 = np.expand_dims(fmri_cond_1[r], 0)
                    test_cond_2 = np.expand_dims(fmri_cond_2[r], 0)
                    dist_test = test_cond_1  - test_cond_2

                    cv_dist = np.dot(dist_train, np.linalg.inv(cov_train))[0][0] #formula for squared cv euclidean distance?
                    cv_distances[r] = cv_dist

            
            if distance_type in ['classification', 'dv-classification']:
                y_train = np.zeros(((fmri_cond_1.shape[0]-1)*2))
                y_train[int(len(y_train)/2):] = 1
                y_test = np.asarray((0, 1))
                scores = np.zeros(fmri_cond_1.shape[0])

                for r in range(fmri_cond_1.shape[0]):

                    # Define the training/test partitions
                    X_train = np.append(np.delete(fmri_cond_1, r, 0),
                        np.delete(fmri_cond_2, r, 0), 0)
                    X_test = np.append(np.expand_dims(fmri_cond_1[r], 0),
                        np.expand_dims(fmri_cond_2[r], 0), 0)
                    
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
    print(f'rdm for {roi} done ')
    print('in {:.4f} seconds'.format(toc-tic))

    return (roi, rdm_array)
