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
parser.add_argument('--data_split', default='training', type=str)
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--eeg_distance_type', default='pearson', type=str)
parser.add_argument('--fmri_distance_type', default='pearson', type=str)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--sfreq', default=500, type=int)
parser.add_argument('--eval_method', default='spearman', type=str)
parser.add_argument('--model_metric', default='euclidean', type=str)
parser.add_argument('--ceiling', default=1, type=int)
parser.add_argument('--roi', type=str)

args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

sfreq_format = format(args.sfreq, '04')

####################### Load data #######################################
upper_folder = f'/scratch/azonneveld/rsa/fusion/eeg-fmri/{args.data_split}/z_{args.zscore}/{args.roi}/fmri_{args.fmri_distance_type}/eeg_{args.eeg_distance_type}/{args.eval_method}/' 
file_path = upper_folder + f'cors.pkl'

with open(file_path, 'rb') as f: 
    upper_data = pickle.load(f)
upper_lim = upper_data['cor_values']


features = ['objects', 'scenes', 'actions']
feature_cors_dict = {
    'objects': [],
    'scenes': [],
    'actions': []
}
feature_ps_dict = {
    'objects': [],
    'scenes': [],
    'actions': []
}

for feature in features:
    fusion_file = f'/scratch/azonneveld/rsa/fusion/triple/{args.data_split}/{args.roi}/{feature}/fmri_{args.fmri_distance_type}/eeg_{args.eeg_distance_type}/vp_results_{args.eval_method}.pkl'
    # fusion_file = f'/scratch/azonneveld/rsa/fusion/triple/test/EBA/{feature}/fmri_pearson/eeg_pearson/vp_results_spearman.pkl'
    with open(fusion_file, 'rb') as f: 
        fusion_data = pickle.load(f)
    fusion_cors = fusion_data['results'][0]
    fusion_ps = fusion_data['results'][1]
    feature_cors_dict[feature] = fusion_cors
    feature_ps_dict[feature] = fdrcorrection(fusion_ps, alpha=0.05)[1]

######################### Plot ####################################
max_cors = []
for i in range(len(features)):
    feature = features[i]
    feature_cors = feature_cors_dict[feature]
    max_cor = np.max(feature_cors)
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
fig, ax = plt.subplots(dpi=400)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    stats_df = pd.DataFrame()
    stats_df['cors'] = feature_cors_dict[feature]
    stats_df['ps'] = np.array(feature_ps_dict[feature]) < 0.05
    # stats_df['ps'] = np.array(ps[feature]) < alpha
    stats_df['upper'] = upper_lim
    stats_df['times'] = fusion_data['times']
    
    y_limit = height_constant * ga_max
    format_ps = [y_limit if i == True else np.nan for i in stats_df['ps']]
    stats_df['format_ps'] = format_ps

    ax.plot(stats_df['times'], stats_df['cors'], label=feature, color=colour)
    ax.plot(stats_df['times'], stats_df['format_ps'], 'ro', color=colour, markersize=3)

    height_constant = height_constant - 0.1

if args.ceiling == True:
    ax.fill_between(stats_df['times'], stats_df['upper'], 0, color='gray', alpha=.2, label='fusion')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title(f'model-eeg-fmri {args.roi}')
# ax.set_title(f'model-eeg-fmri EBA')
ax.set_xlabel('Time (s)')
ax.set_ylabel(f'Variance explained (%)')
ax.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

res_folder =  f'/scratch/azonneveld/rsa/fusion/triple/plots/{args.data_split}/{args.roi}/fmri_{args.fmri_distance_type}/eeg_{args.eeg_distance_type}/ceiling_{args.ceiling}/'
# res_folder =  f'/scratch/azonneveld/rsa/fusion/triple/plots/test/EBA/{feature}/fmri_pearson/eeg_pearson/'
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)

# img_path = res_folder + f'vp_results_spearman.png'
img_path = res_folder + f'vp_results_{args.eval_method}.png'
plt.savefig(img_path)
plt.clf()

print('Done plotting')