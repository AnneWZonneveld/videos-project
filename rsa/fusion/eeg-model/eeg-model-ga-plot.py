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
from statsmodels.stats.multitest import fdrcorrection
import argparse

# Take arguments
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--distance_type', default='euclidean-cv', type=str)
parser.add_argument('--bin_width', default=0, type=float)
parser.add_argument('--slide', default=0, type=float)

args = parser.parse_args()


####################### Load data #######################################
sub_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/standard/GA/{args.distance_type}/bin_{args.bin_width}/slide_{args.slide}/' 
res_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/standard/plots/GA/'

if not os.path.exists(res_folder) == True:
    os.mkdir(res_folder)

features = ['objects', 'scenes', 'actions']

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
    
    ps[feature] = fdrcorrection(ps[feature], alpha=args.alpha)[1]
    # ps[feature] = fdrcorrection(ps[feature], alpha=alpha)[1]


######################### Plot ####################################

max_cors = []
for i in range(len(features)):
    feature = features[i]
    feature_cors = cors[feature]
    max_cor = np.max(feature_cors)
    max_cors.append(max_cor)

ga_max = np.max(max_cors)
height_constant = 1.5

colours = ['b', 'r', 'g']
fig, ax = plt.subplots(dpi=300)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    stats_df = pd.DataFrame()
    stats_df['cors'] = cors[feature]
    stats_df['ps'] = np.array(ps[feature]) < args.alpha
    # stats_df['ps'] = np.array(ps[feature]) < alpha
    stats_df['lower_CI'] = cis_low[feature]
    stats_df['upper_CI'] = cis_up[feature]
    stats_df['times'] = cor_res['times']
    
    y_limit = height_constant * ga_max
    format_ps = [y_limit if i == True else np.nan for i in stats_df['ps']]
    stats_df['format_ps'] = format_ps

    ax.plot(stats_df['times'], stats_df['cors'], label=feature, color=colour)
    ax.fill_between(stats_df['times'], stats_df['lower_CI'], stats_df['upper_CI'], color=colour, alpha=.1)
    ax.plot(stats_df['times'], stats_df['format_ps'], 'ro', color=colour, markersize=3)

    height_constant = height_constant - 0.1

ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
ax.set_title(f'EEG-model correlation GA')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Spearman cor')
ax.legend()
fig.tight_layout()

img_path = res_folder + f'var_explained_{args.distance_type}_{args.bin_width}.png'
plt.savefig(img_path)
plt.clf()