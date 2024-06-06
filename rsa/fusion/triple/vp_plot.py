"""
Plotting commonality score results. 
Group level.

Parameters
----------
data_split: str
    Train or test. 
alpha: float
    Significane threshold.
eeg_distance_type: str
    Whether to use EEG RDMs based on 'euclidean'/'pearson'
fmri_distance_type: str
    Whether to use fMRI RDMs based on 'euclidean'/'pearson'
model_metric: str
    Metric used in the model RDM; 'pearson'/'euclidean'
eval_method: str
    Method to compute similarity between RDMs; 'spearman' or 'pearson'
ceiling: int
    Plot eeg-fmri fusion (aka ceiling)  results y/n
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
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--eeg_distance_type', default='pearson', type=str)
parser.add_argument('--fmri_distance_type', default='pearson', type=str)
parser.add_argument('--eval_method', default='spearman', type=str)
parser.add_argument('--model_metric', default='euclidean', type=str)
parser.add_argument('--ceiling', default=0, type=int)
parser.add_argument('--roi', default='V1v', type=str)

args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


####################### Load data #######################################
upper_folder = f'/scratch/azonneveld/rsa/fusion/eeg-fmri/{args.data_split}/z_1/{args.roi}/fmri_{args.fmri_distance_type}/eeg_{args.eeg_distance_type}/{args.eval_method}/' 
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
    with open(fusion_file, 'rb') as f: 
        fusion_data = pickle.load(f)
    fusion_cors = fusion_data['results'][0]
    fusion_ps = fusion_data['results'][1]
    feature_cors_dict[feature] = fusion_cors
    feature_ps_dict[feature] = multipletests(fusion_ps, alpha=args.alpha, method='bonferroni')[1]

######################### Plot ####################################
    
if args.ceiling == True:
     ga_max = np.max(upper_lim)
else:
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
                rc={'font.size': 16, 
                    'xtick.labelsize': 14, 
                    'ytick.labelsize':14, 
                    'axes.titlesize' : 13,
                    'figure.titleweight': 'bold', 
                    'axes.labelsize': 15, 
                    'legend.fontsize': 12, 
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
    stats_df['ps'] = feature_ps_dict[feature]
    stats_df['sign'] = np.array(feature_ps_dict[feature]) < args.alpha
    stats_df['times'] = fusion_data['times']

    if args.ceiling == True:
        stats_df['upper'] = upper_lim
    
    y_limit = height_constant * ga_max
    format_ps = [y_limit if i == True else np.nan for i in stats_df['sign']]
    stats_df['format_ps'] = format_ps

    ax.plot(stats_df['times'], stats_df['cors'], label=feature, color=colour)
    ax.plot(stats_df['times'], stats_df['format_ps'], 'ro', color=colour, markersize=3)

    height_constant = height_constant - 0.1

    # Print stats
    onset_t = np.nan
    onset_cor = np.nan
    for j in range(len(stats_df)):
        if stats_df['sign'].iloc[j] == True:
            onset_t = stats_df['times'].iloc[j]
            onset_cor = stats_df['cors'].iloc[j]
            break
    try:
        print("########################################")
        print(f'Feature {feature} sign onset: {onset_t}')
        print(f'Cor : {onset_cor}')
        print("########################################")
    except:
        print('No sign timepoints')

    cor_16 = feature_cors_dict[feature][16]
    t_16 = fusion_data['times'][16]

    print("########################################")
    print(f'Feature {feature} ~120 ms peak: {t_16}')
    print(f'Cor : {cor_16}')
    print("########################################")

    cor_23 = feature_cors_dict[feature][23]
    t_23 = fusion_data['times'][23]

    print("########################################")
    print(f'Feature {feature} ~260 ms peak: {t_23}')
    print(f'Cor : {cor_23}')
    print("########################################")

    max_cor = np.max(feature_cors_dict[feature])
    max_idx = np.where(feature_cors_dict[feature] == max_cor)
    max_t = fusion_data['times'][max_idx]

    print("########################################")
    print(f'Feature {feature} peak: {max_t}')
    print(f'Cor : {max_cor}')
    print("########################################")

    cor_35 = feature_cors_dict[feature][35]
    t_35 = fusion_data['times'][35]

    print("########################################")
    print(f'Feature {feature} second peak: {t_35}')
    print(f'Cor : {cor_35}')
    print("########################################")

if args.ceiling == True:
    ax.fill_between(stats_df['times'], stats_df['upper'], 0, color='gray', alpha=.2, label='fusion')

ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title(f'model-eeg-fmri {args.roi}')
ax.set_xlabel('Time (s)')
ax.set_ylabel(f'Unique variance explained (%)')
ax.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

res_folder =  f'/scratch/azonneveld/rsa/fusion/triple/plots/{args.data_split}/{args.roi}/fmri_{args.fmri_distance_type}/eeg_{args.eeg_distance_type}/ceiling_{args.ceiling}/'
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)

img_path = res_folder + f'vp_results_{args.eval_method}_{args.roi}_{args.data_split}.png'
plt.savefig(img_path)
plt.clf()

print('Done plotting')


######################### Plot: relative unique variance ####################################
print('Plotting relative unique variance explained')

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


features = ['objects', 'scenes', 'actions']
colours = ['b', 'r', 'g']

fig, axes = plt.subplots(dpi=300)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    stats_df = pd.DataFrame()
    stats_df['values'] = (np.array([0 if upper_lim[i] < 0.005 * np.max(upper_lim) else feature_cors_dict[feature][i] for i in range(len(upper_lim))])/ upper_lim)*100
    stats_df['times'] = fusion_data['times']

    sns.lineplot(data=stats_df, x='times', y='values', label=feature, color=colour)

axes.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
axes.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
axes.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes.set_title(f'model-eeg-fmri GA')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Proportion variance exlained (%)')
axes.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

res_folder =  f'/scratch/azonneveld/rsa/fusion/triple/plots/{args.data_split}/{args.roi}/fmri_{args.fmri_distance_type}/eeg_{args.eeg_distance_type}/'
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)

img_path = res_folder + f'relative_fusion_vp.png'
plt.savefig(img_path)
plt.clf()


######################### Plot: cummulative variance ####################################
print('Plotting cum unique variance explained')

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


features = ['objects', 'scenes', 'actions']
colours = ['b', 'r', 'g']

fig, axes = plt.subplots(dpi=300)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    stats_df = pd.DataFrame()
    stats_df['cors'] = feature_cors_dict[feature]
    stats_df['times'] = fusion_data['times']
    y_cum = np.cumsum(stats_df['cors'])
    stats_df['values'] = y_cum

    sns.lineplot(data=stats_df, x='times', y='values', label=feature, color=colour)

axes.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
axes.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
axes.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes.set_title(f'model-eeg-fmri GA')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Cumalitve unique variance exlained (%)')
axes.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

res_folder =  f'/scratch/azonneveld/rsa/fusion/triple/plots/{args.data_split}/{args.roi}/fmri_{args.fmri_distance_type}/eeg_{args.eeg_distance_type}/'
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)

img_path = res_folder + f'cumaltive_fusion_vp.png'
plt.savefig(img_path)
plt.clf()