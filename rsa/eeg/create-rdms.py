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
import tensorflow_hub as hub
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.svm import SVC

# Parameters
sub = '01'
zscore = True
data_split = 'train'
distance_type = 'euclidean-cv'
weighted_class = True

print(f'Running for sub-{sub}, {data_split}, {distance_type}')

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


################################### Load EEG data ######################################
data_dir = f'/scratch/giffordale95/projects/eeg_videos/dataset/preprocessed_data/dataset_02/eeg/sub-{sub}/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-0050/preprocessed_data.npy'
data_dict = np.load(data_dir, allow_pickle=True).item()
times = data_dict['times']
ch_names = data_dict['ch_names']
info = data_dict['info']
eeg_data_list = data_dict['eeg_data']
stimuli_presentation_order_list = data_dict['stimuli_presentation_order']

# Z score data per session
for session in range(len(eeg_data_list)):
    data_shape = eeg_data_list[session].shape
    data_provv = np.reshape(eeg_data_list[session], (len(eeg_data_list[session]), -1))
    if zscore == True:
        scaler = StandardScaler()
        data_provv = scaler.fit_transform(data_provv)
    data_provv = np.reshape(data_provv, data_shape)
    if session == 0:
        eeg_data = data_provv
        stimuli_presentation_order = stimuli_presentation_order_list[session]
    else:   
        eeg_data = np.append(eeg_data, data_provv, 0)
        stimuli_presentation_order = np.append(stimuli_presentation_order, stimuli_presentation_order_list[session], 0)
    del data_provv
del eeg_data_list, stimuli_presentation_order_list

# Select the data split
if data_split == 'train':
    idx = np.where(stimuli_presentation_order <= 1000)[0]
elif data_split == 'test':
    idx = np.where(stimuli_presentation_order > 1000)[0]
eeg_data = eeg_data[idx]
stimuli_presentation_order = stimuli_presentation_order[idx]


################################### Constructing RDMs #############################################
# Results array of shape:
# (Video conditions × Video conditions × EEG time points)

if distance_type != 'euclidean':
    n_perm = 3
else:
    n_perm = 1

video_conditions = np.unique(stimuli_presentation_order)
n_conditions = video_conditions.shape[0]
n_channels = len(ch_names)
n_time = times.shape[0]
rdms_array = np.zeros((n_conditions, n_conditions, eeg_data.shape[2]), dtype='float32')


# Loop over EEG time points, image-conditions and EEG repetitions
for t in tqdm(range(eeg_data.shape[2])):
# for t in tqdm(range(2)):

    if distance_type != 'euclidean':
        permutation_array = np.zeros((n_perm, n_conditions, n_conditions), dtype='float32')

    for p in tqdm(range(n_perm)):

        for v1 in range(n_conditions):
            for v2 in range(v1):
                idx_1 = resample(np.where(stimuli_presentation_order == video_conditions[v1])[0],replace=False)
                idx_2 = resample(np.where(stimuli_presentation_order == video_conditions[v2])[0],replace=False)
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
                    distance = np.linalg.norm(eeg_cond_1-eeg_cond_2).astype('float32')

                    rdms_array[v1, v2, t] = distance
                    rdms_array[v2, v1, t] = distance

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

                    permutation_array[p, v1, v2] = distance
                    permutation_array[p, v2, v1] = distance

                
                if distance_type == 'classification':
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
                        if weighted_class == True:
                            # Weighting of decision values (distance to hyper plane / norm weights), as in gugguenmos et al., 2018
                            decision_values = dec_svm.decision_function(X_test) / np.linalg.norm(dec_svm.coef_)
                            weighted_score = sum(decision_values *((y_pred == y_test) - 0.5))/ len(y_test)
                            scores[r] = weighted_score
                        else: 
                            scores[r]= sum((y_pred == y_test) - 0.5)/ len(y_test)
                            # scores[r] = sum(y_pred == y_test) / len(y_test)

                    score = np.mean(scores).astype('float32')
                    
                    permutation_array[p, v1, v2] = score
                    permutation_array[p, v2, v1] = score

    if distance_type != 'euclidean':
        rdms_array[:, :, t] = np.mean(permutation_array, axis=0).astype('float32')    


# Save results
results_dict = {
    'n_perm': n_perm,
    'data_split': data_split, 
    'sub': sub, 
    'zscore': zscore,
    'distance_type': distance_type,
    'weighted_class': weighted_class,
	'rdms_array': rdms_array,
	'times': times,
	'ch_names': ch_names,
	'info': info
}

res_folder = f'/scratch/azonneveld/rsa/eeg/rdms/sub-{sub}/' 
file_path = res_folder + f'{distance_type}_{data_split}.pkl'

# Save all model rdms
with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)





