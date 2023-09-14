""" File to explore meta data BOLD moments"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
import scipy
import gensim.downloader as api
from gensim.models import KeyedVectors

# Load json file to df
file_path = '/scratch/giffordale95/projects/eeg_videos/videos_metadata/annotations.json'
md = pd.read_json(file_path).transpose()

# Set image folder path
img_folder = '/scratch/azonneveld/meta-explore'

# Inspect data structure
md.head()
col_names = md.columns.values.tolist()
n_trials = md.shape[0]

# Create count df for different label types 
vars_oi = ['objects', 'scenes', 'actions']
vars_dict = {}

for var in vars_oi:
    temp_df = md[var].to_numpy()
    all_labels = np.empty(5*n_trials, dtype=object)

    for i in range(5*n_trials):
        trial = i // 5
        j = i % 5 
        all_labels[i] = temp_df[trial][j]
    
    label_unique, label_counts = np.unique(all_labels, return_counts=True)
    label_unique = np.expand_dims(label_unique, axis = 1)
    label_counts = np.expand_dims(label_counts, axis = 1)

    count_df = pd.concat([pd.DataFrame(label_unique), pd.DataFrame(label_counts)], axis=1)
    count_df.columns = ['label', 'count']
    count_df = count_df.sort_values(by='count', axis=0, ascending=False)

    vars_dict[var] = count_df

# unique labels: objects 771, scenes 323, actions 260 --> biased comparison?

# Plot label counts
fig, ax = plt.subplots(1,3, dpi = 200)
for i in range(len(vars_oi)):
    var = vars_oi[i]

    sns.barplot(x='label', y='count', data=vars_dict[var], ax=ax[i])
    ax[i].set_title(f'{var}')
    ax[i].set_xticks([])
    ax[i].set_xlabel('Labels')

fig.tight_layout()
img_path = img_folder + '/label-hist.png'
plt.savefig(img_path)
plt.clf()
# Highly skewed distirubions

shell()

# Load word model 
# model = api.load('word2vec-google-news-300')
vectors = KeyedVectors.load('vectors.kv')
