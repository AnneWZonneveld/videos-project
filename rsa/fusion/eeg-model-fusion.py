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
from vp_utils import corr_nullDist

n_subs = 3
data_split = 'train'
distance_type = 'euclidean-cv'
feature = 'objects' 
alpha = 0.05

res_folder = '/scratch/azonneveld/rsa/fusion/plots/'

# Load in model rdms
print('Loading model rdms')
model_file = '/scratch/azonneveld/rsa/model/rdms/t2/rdm_t2_freq.pkl' #change this to permuted freq matrix
with open(model_file, 'rb') as f: 
    model_rdms = pickle.load(f)
model_rdms = model_rdms[data_split]

# Load in EEG rdms for all participants
print('Loading neural rdms')
rdms_array = np.zeros((n_subs, 1000, 1000, 185))
for i in range(n_subs):

    sub_format = format(i + 1, '02')
    eeg_file = f'/scratch/azonneveld/rsa/eeg/rdms/sub-{sub_format}/avg_{distance_type}.pkl'

    with open(eeg_file, 'rb') as f: 
        eeg_rdms = pickle.load(f)
    
    rdms_array[i, :, :, :] = eeg_rdms['rdms_array']

# Average RDMs over participants
GA_rdms = np.mean(rdms_array, axis=0)
# times = range(GA_rdms.shape[2])
times = 50 #test


# Calculate correlation
print('Calculating rdm correlation')
feature_rdm = model_rdms[feature]
rdm_cors = []
rdm_cor_ps = []

for t in tqdm(range(times)): #could do multiprocessing here?

    # Calc correlation
    neural_rdm = GA_rdms[:, :, t]
    rdm_cor = spearmanr(squareform(neural_rdm, checks=False), squareform(feature_rdm, checks=False))[0] 
    rdm_cors.append(rdm_cor)

    # Calc significance
    rdm_p = corr_nullDist(rdm_cor, neural_rdm, feature_rdm, its=10) # testing purpose
    rdm_cor_ps.append(rdm_p)

stats_df = pd.DataFrame()
stats_df['cors'] = rdm_cors
stats_df['ps'] = np.array(rdm_cor_ps) < alpha
stats_df['times'] = eeg_rdms['times']


# Plot 
max_cor = np.max(stats_df['cors'])
y_limit = 1.5 * max_cor
formated_ps = [y_limit for i in stats_df['ps'] if i == True else np.nan]

fig, ax = plt.subplots()
sns.lineplot(data=stats_df, x='times', y='cors', label=feature)
plt.plot(times, formated_ps, 'ro')
ax.set_title('EEG-model correlation')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Spearman cor')

img_path = res_folder + f'var_explained_test.png'
plt.savefig(img_path)
plt.clf()

# Bootstrap confidence interval

# Asses in comparison to noise ceiling
    # Calc noise ceiling

