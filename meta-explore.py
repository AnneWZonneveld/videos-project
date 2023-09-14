""" File to explore meta data BOLD moments"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
import scipy
import gensim
from gensim.models import KeyedVectors

# Load json file to df
file_path = '/scratch/giffordale95/projects/eeg_videos/videos_metadata/annotations.json'
md = pd.read_json(file_path).transpose()

# Set image folder path
img_folder = '/scratch/azonneveld/meta-explore'

# Inspect data structure
md.head()
col_names = md.columns.values.tolist()
total_n_trials = md.shape[0]

# Create count df for different label types train and test set
zets = ['train', 'test']
vars_oi = ['objects', 'scenes', 'actions']

zets_dict = {}
for zet in zets:
    temp_zet = md[md['set']==zet]
    n_trials = temp_zet.shape[0]
    vars_dict = {}

    for var in vars_oi:
        temp_df = temp_zet[var].to_numpy()
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

    zets_dict[zet] = vars_dict

# Descriptives 
zets_max = {}
for zet in zets:
    vars_dict = zets_dict[zet]
    labels_max = []

    for var in vars_oi:
        total_n = vars_dict[var].shape[0]
        max = vars_dict[var]['count'].iloc[0]
        labels_max.append(max)
        print(f'{zet} {var} unique labels: {total_n}')
        print(f'{zet} {var} top 10:')
        top_10 = vars_dict[var].iloc[0:10]
        print(f'{top_10}')

    zets_max[zet] = np.max(np.array(labels_max))

# Plot all label counts
fig, ax = plt.subplots(2,3, dpi = 300, sharey='row')
for j in range(len(zets)):
    zet = zets[j]
    vars_dict = zets_dict[zet]

    for i in range(len(vars_oi)):
        var = vars_oi[i]
        total_n = vars_dict[var].shape[0]

        sns.barplot(x='label', y='count', data=vars_dict[var], ax=ax[j, i])
        ax[j, i].set_title(f'{zet} {var}: {total_n}')
        ax[j, i].set_xticks([])
        ax[j, i].set_xlabel('Labels')

fig.tight_layout()
img_path = img_folder + '/label-hist.png'
plt.savefig(img_path)
plt.clf()

# Plot top 15 label counts
fig, ax = plt.subplots(2,3, dpi = 200, sharex='row')
fig.set_figwidth(12)
for j in range(len(zets)):
    zet = zets[j]
    vars_dict = zets_dict[zet]

    for i in range(len(vars_oi)):
        var = vars_oi[i]
        total_n = vars_dict[var].shape[0]

        sns.barplot(x='count', y='label', data=vars_dict[var].iloc[0:15], ax=ax[j, i])
        ax[j, i].set_xlabel('count')
        ax[j, i].set_ylabel('')
        ax[j,i].set_title(f'{zet} {var}: {total_n}')
        ax[j,i].tick_params(axis='y', labelsize=6)

fig.tight_layout()
img_path = img_folder + '/label-hist-top.png'
plt.savefig(img_path)
plt.clf()

# ----------------------------------------------------------

shell()

# Load word model 
w2v_vectors = gensim.downloader.load('word2vec-google-news-300')

#Test
w2v_vectors.most_similar('twitter')
w2v_vectors['twitter'].shape
w2v_vectors.distance("media", "twitter")

# Get label vectors (works for scenes/actions but not for objects) --> what is good word model? which corpus
# Actions has some odd labels (like adult+female+singing)
# Object labels highly specific
# Some object labels are like scences: 'toy store'
labels = np.unique(vars_dict['scenes']['label'])
labels_vector = []
w2v_vectors[labels[301]]
