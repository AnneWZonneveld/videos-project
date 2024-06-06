"""
Plotting commonality score results with control 
(aka when variables for X and Y are swapped in commonality calculation).
Group level.


Parameters
----------
data_split: str
    Train or test. 
alpha: float
    Significane threshold.
roi: str
    Name ROI.

"""



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
from statsmodels.stats.multitest import fdrcorrection, multipletests
import argparse

# Take arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_split', default='training', type=str)
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--roi', type=str)

args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

####################### Load data #######################################

feature_cors_dict = {}
feature_ps_dict = {}
feature_cors_control_dict= {}
feature_ps_control_dict = {}

features = ['objects', 'scenes', 'actions']
for feature in features:
    fusion_file = f'/scratch/azonneveld/rsa/fusion/triple/test/{args.roi}/{feature}/fmri_pearson/eeg_pearson/vp_results_spearman.pkl'
    control_file = f'/scratch/azonneveld/rsa/fusion/triple/test/PPA/{args.roi}/fmri_pearson/eeg_pearson/vp_results_spearman_control.pkl'
    with open(fusion_file, 'rb') as f: 
        fusion_data = pickle.load(f)
    with open(fusion_file, 'rb') as f: 
        control_data = pickle.load(f)
    fusion_cors = fusion_data['results'][0]
    fusion_ps = fusion_data['results'][1]
    feature_cors_dict[feature] = fusion_cors
    feature_ps_dict[feature] = multipletests(fusion_ps, alpha=0.05, method='bonferroni')[1]
    cors_control = control_data['results'][0]
    ps_control = control_data['results'][1]
    feature_cors_control_dict[feature] = cors_control
    feature_ps_control_dict[feature] = multipletests(ps_control, alpha=0.05, method='bonferroni')[1]

######################### Plot ####################################
max_cors = []
for i in range(len(features)):
    feature = features[i]
    feature_cors = feature_cors_dict[feature]
    max_cor = np.max(feature_cors)
    feature_cors_control = feature_cors_control_dict[feature]
    max_cor = np.max(feature_cors_control)
    max_cors.append(max_cor)

ga_max = np.max(max_cors)
height_constant = 1.5

sns.set_style('white')
sns.set_style("ticks")
sns.set_context('paper', 
                rc={'font.size': 14, 
                    'xtick.labelsize': 10, 
                    'ytick.labelsize':10, 
                    'axes.titlesize' : 13,
                    'figure.titleweight': 'bold', 
                    'axes.labelsize': 13, 
                    'legend.fontsize': 8, 
                    'font.family': 'Arial',
                    'axes.spines.right' : False,
                    'axes.spines.top' : False})

colours = ['b', 'r', 'g']
control_colours = ['purple', 'orange', 'yellow']
fig, ax = plt.subplots(dpi=400)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    stats_df = pd.DataFrame()
    stats_df['cors'] = feature_cors_dict[feature]
    stats_df['ps'] = np.array(feature_ps_dict[feature]) < 0.05
    stats_df['times'] = fusion_data['times']
    
    y_limit = height_constant * ga_max
    format_ps = [y_limit if i == True else np.nan for i in stats_df['ps']]
    stats_df['format_ps'] = format_ps

    ax.plot(stats_df['times'], stats_df['cors'], label=feature, color=colour)
    ax.plot(stats_df['times'], stats_df['format_ps'], 'ro', color=colour, markersize=3)

    height_constant = height_constant - 0.1

    control_colour = control_colours[i]
    stats_df = pd.DataFrame()
    stats_df['cors'] = feature_cors_control_dict[feature]
    stats_df['ps'] = np.array(feature_ps_control_dict[feature]) < 0.05
    stats_df['times'] = fusion_data['times']
    
    y_limit = height_constant * ga_max
    format_ps = [y_limit if i == True else np.nan for i in stats_df['ps']]
    stats_df['format_ps'] = format_ps

    ax.plot(stats_df['times'], stats_df['cors'], label=feature, color=control_colour)
    ax.plot(stats_df['times'], stats_df['format_ps'], 'ro', color=control_colour, markersize=3)

    height_constant = height_constant - 0.1

ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title(f'model-eeg-fmri PPA')
ax.set_xlabel('Time (s)')
ax.set_ylabel(f'Variance explained (%)')
ax.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

res_folder =  f'/scratch/azonneveld/rsa/fusion/triple/plots/test/{args.roi}/fmri_pearson/eeg_pearson/'
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)

img_path = res_folder + f'vp_results_spearman_control.png'
plt.savefig(img_path)
plt.clf()

print('Done plotting')