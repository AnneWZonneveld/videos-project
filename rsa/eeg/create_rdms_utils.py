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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.svm import SVC
import argparse
from functools import partial
from multiprocessing import shared_memory
from multiprocessing import current_process, cpu_count
import time


def compute_rdm(t, data_shape, pseudo_order, shm, data_split='train', distance_type='euclidean'):

    print(f'compute rdm for t={t}')
    tic = time.time()

    if data_split=='train':
        n_conditions = 1000
    elif data_split=='test':
        n_conditions == 102

    existing_shm = shared_memory.SharedMemory(name=shm)
    eeg_data = np.ndarray(data_shape, dtype='float32', buffer=existing_shm.buf)
    
    rdm_array = np.zeros((n_conditions, n_conditions))
    
    combination = 0 
    for v1 in tqdm(range(n_conditions)):
        for v2 in tqdm(range(v1)):
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
            if data_split == 'test':
                n_ptrials_repeats = 4
            elif data_split == 'train':
                n_ptrials_repeats = 2
            n_pseudo_trials = int(np.ceil(len(eeg_cond_1) / n_ptrials_repeats))
            pseudo_data_1 = np.zeros((n_pseudo_trials, eeg_cond_1.shape[1]))
            pseudo_data_2 = np.zeros((n_pseudo_trials, eeg_cond_2.shape[1]))

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
                # distance = np.linalg.norm(eeg_cond_1-eeg_cond_2).astype('float32')
                distance = np.dot((eeg_cond_1 - eeg_cond_2), np.transpose(eeg_cond_1 - eeg_cond_2)).astype('float32')[0][0] 

                rdm_array[v1, v2] = distance
                rdm_array[v2, v1] = distance

                del eeg_cond_1, eeg_cond_2
            
            if distance_type == 'euclidean-cv':
                cv_distances = np.zeros(len(eeg_cond_1))

                for r in range(len(eeg_cond_1)):

                    # Define the training/test partitions (LOOCV)
                    train_cond_1 = np.delete(eeg_cond_1, r, 0)
                    train_cond_2 = np.delete(eeg_cond_2, r, 0)
                    dist_train = np.expand_dims(np.mean(train_cond_1, 0) - np.mean(train_cond_2, 0), 0)
                    
                    test_cond_1 = np.expand_dims(eeg_cond_1[r], 0)
                    test_cond_2 = np.expand_dims(eeg_cond_2[r], 0)
                    dist_test = test_cond_1  - test_cond_2

                    cv_dist = np.dot(dist_train, np.transpose(dist_test)).astype('float32')[0][0] #formula for squared cv euclidean distance?
                    cv_distances[r] = cv_dist
                
                distance = np.mean(cv_distances, dtype='float32') 

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

                score = np.mean(scores).astype('float32')
                
                rdm_array[v1, v2] = score
                rdm_array[v2, v1] = score

            combination = combination + 1
    
    toc = time.time()
    print(f'rdm for t={t} done ')
    print('timepoint done in {:.4f} seconds'.format(toc-tic))

    return (rdm_array, t)





