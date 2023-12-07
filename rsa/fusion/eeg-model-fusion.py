import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, AgglomerativeClustering
import tensorflow as tf
import tensorflow_hub as hub
from datasets import load_dataset, Dataset, DatasetDict
from natsort import index_natsorted
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from functools import partial
import multiprocessing as mp
from vp_utils import *

n_subs = 3
data_split = 'train'
distance_type = 'euclidean-cv'
feature = 'objects' 
alpha = 0.05
n_cpus = 2

res_folder = '/scratch/azonneveld/rsa/fusion/plots/'

########################### Loading in data #####################################

print('Loading model rdms')
model_file = '/scratch/azonneveld/rsa/model/rdms/t2/rdm_t2_freq.pkl' #change this to permuted freq matrix
with open(model_file, 'rb') as f: 
    model_rdms = pickle.load(f)
model_rdms = model_rdms[data_split]

print('Loading neural rdms')
rdms_array = np.zeros((n_subs, 1000, 1000, 185))
for i in range(n_subs):

    sub_format = format(i + 1, '02')
    eeg_file = f'/scratch/azonneveld/rsa/eeg/rdms/sub-{sub_format}/avg_{distance_type}.pkl'

    with open(eeg_file, 'rb') as f: 
        eeg_rdms = pickle.load(f)
    
    rdms_array[i, :, :, :] = eeg_rdms['rdms_array']

############################# Analysis ###########################################

# Average RDMs over participants
GA_rdms = np.mean(rdms_array, axis=0)
# times = range(GA_rdms.shape[2])
times = range(3) #test

# Calculate correlation
print('Calculating rdm correlation')
feature_rdm = model_rdms[feature]

partial_calc_rsquared = partial(calc_rsquared,
                                GA_rdms = GA_rdms,
                                feature_rdm = feature_rdm,
                                feature=feature,
                                n_cpus=n_cpus,
                                its=10)


pool = mp.Pool(n_cpus)
results = pool.map(partial_calc_rsquared, times)
pool.close()
print('Done calculating correlations')

rdm_cors = []
rdm_cor_ps = []
for i in len(results):
    rdm_cors.append(results[i][0])
    rdm_cor_ps.append(results[i][1])


# Bootstrap confidence interval
print('Bootstrapping CI')
partial_cor_variability = partial(cor_variability,
                                GA_rdms = GA_rdms,
                                feature_rdm = feature_rdm,
                                its=10)

pool = mp.Pool(n_cpus)
results = pool.map(partial_cor_variability, times)
pool.close()
print('Done bootstrapping CI')

lower_CI = []
upper_CI = []
for i in len(results):
    lower_CI.append(results[i][0])
    upper_CI.append(results[i][1])


# Plot 
print('Creating plots')
stats_df = pd.DataFrame()
stats_df['cors'] = rdm_cors
stats_df['ps'] = np.array(rdm_cor_ps) < alpha
stats_df['lower_CI'] = lower_CI
stats_df['upper_CI'] = upper_CI
# stats_df['times'] = eeg_rdms['times']
stats_df['times'] = times

max_cor = np.max(stats_df['cors'])
y_limit = 1.5 * max_cor
format_ps = [y_limit if i == True else np.nan for i in stats_df['ps']]
stats_df['format_ps'] = format_ps

fig, ax = plt.subplots()
ax.plot(stats_df['times'], stats_df['cors'], label='feature', color='b')
ax.fill_between(stats_df['times'], stats_df['lower_CI'], stats_df['upper_CI'], color='b', alpha=.1)
ax.plot(stats_df['times'], stats_df['format_ps'], 'ro', color='b')
ax.set_title('EEG-model correlation')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Spearman cor')
ax.legend()
fig.tight_layout()

img_path = res_folder + f'var_explained_test.png'
plt.savefig(img_path)
plt.clf()


# Asses in comparison to noise ceiling
    # Calc noise ceiling

