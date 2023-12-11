import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import argparse

# Take arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--alpha', default=0.05, type=float)

args = parser.parse_args()


####################### Load data #######################################
sub_format = format(args.sub, '02')
sub_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/sub-{sub_format}/' 
res_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/plots/'

# features = ['objects', 'scenes', 'actions']
features = ['scenes', 'actions']

cors = {}
ps = {}
cis_low = {}
cis_up = {}

for feature in features:
    cors[feature] = []
    ps[feature] = []
    cis_low[feature] = []
    cis_up[feature] = []  

for feature in features:

    cor_path = sub_folder + f'cors_{feature}.pkl'
    ci_path = sub_folder + f'cis_{feature}.pkl'

    with open(cor_path, 'rb') as f: 
        cor_res = pickle.load(f)
    with open(ci_path, 'rb') as f: 
        ci_res = pickle.load(f)

    cor_values = cor_res['cor_values']
    cis_values =  ci_res['cis_values']
    for i in range(len(cor_values)):
        cors[feature].append(cor_values[i][0])
        ps[feature].append(cor_values[i][1])
        cis_low[feature].append(cis_values[i][0])
        cis_up[feature].append(cis_values[i][1])


######################### Plot ####################################
colours = ['b', 'r', 'g']
fig, ax = plt.subplots(dpi=300)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    stats_df = pd.DataFrame()
    stats_df['cors'] = cors[feature]
    # stats_df['ps'] = np.array(ps[feature]) < args.alpha
    stats_df['ps'] = np.array(ps[feature]) < alpha
    stats_df['lower_CI'] = cis_low[feature]
    stats_df['upper_CI'] = cis_up[feature]
    stats_df['times'] = cor_res['times']

    max_cor = np.max(stats_df['cors'])
    y_limit = 1.5 * max_cor
    format_ps = [y_limit if i == True else np.nan for i in stats_df['ps']]
    stats_df['format_ps'] = format_ps

    ax.plot(stats_df['times'], stats_df['cors'], label=feature, color=colour)
    ax.fill_between(stats_df['times'], stats_df['lower_CI'], stats_df['upper_CI'], color=colour, alpha=.1)
    ax.plot(stats_df['times'], stats_df['format_ps'], 'ro', color=colour)

ax.set_title('EEG-model correlation')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Spearman cor')
ax.legend()
fig.tight_layout()

img_path = res_folder + f'{sub_format}_var_explained.png'
plt.savefig(img_path)
plt.clf()