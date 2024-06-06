"""
Plot fmri-model fusion results. 
Compare significance results for permutution vs bootstrapped significance method.

Parameters
----------
sub: int
    Subject number.
data_split: str
    Train or test. 
alpha: float
    Significance threshold
distance_type: str
    Whether to use EEG RDMs based on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
eval_method: str
    Method to compute similarity between RDMS; 'spearman' or 'pearson'
model_metric: str
    Metric used in the model RDM; 'pearson'/'euclidean'
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
import argparse
from statsmodels.stats.multitest import fdrcorrection, multipletests

# Take arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_split', default='training', type=str)
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--distance_type', default='euclidean-cv', type=str)
parser.add_argument('--eval_method', default='spearman', type=str)
parser.add_argument('--model_metric', default='euclidean', type=str)

args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


############################ Load data  ############################
features = ['objects', 'scenes', 'actions']     

data_folder = f'/scratch/azonneveld/rsa/fusion/fmri-model/standard/{args.data_split}/model_{args.model_metric}/GA/{args.distance_type}/{args.eval_method}/'
res_folder = f'/scratch/azonneveld/rsa/fusion/fmri-model/plots/standard/{args.data_split}/model_{args.model_metric}/GA/{args.distance_type}/{args.eval_method}/'
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)

all_cors = []
all_ps_random = []
adj_ps_random = []
all_ps_booted = []
adj_ps_booted = []
all_cis_lower = []
all_cis_upper = []
all_features = []
all_rois = []

for feature in features:

    random_cor_file  = data_folder + f'cors_{feature}.pkl'
    with open(random_cor_file, 'rb') as f: 
        random_cor_data = pickle.load(f)
    booted_cor_file  = data_folder + f'cors_{feature}_booted.pkl'
    with open(random_cor_file, 'rb') as f: 
        booted_cor_data = pickle.load(f)
    cis_file = data_folder + f'cis_{feature}.pkl'
    with open(cis_file, 'rb') as f: 
        cis_data = pickle.load(f)
    
    rois = random_cor_data['results'].keys()
    ps_per_feat_random = []
    ps_per_feat_booted = []
    for roi in rois:
        all_cors.append(random_cor_data['results'][roi][0])
        all_ps_random.append(random_cor_data['results'][roi][1])
        all_ps_booted.append(booted_cor_data['results'][roi][1])
        ps_per_feat_random.append(random_cor_data['results'][roi][1])
        ps_per_feat_booted.append(booted_cor_data['results'][roi][1])
        all_cis_lower.append(cis_data['results'][roi][0])
        all_cis_upper.append(cis_data['results'][roi][1])
        all_features.append(feature)
        all_rois.append(roi)
    
    adj_ps_per_feat_random = multipletests(ps_per_feat_random, alpha=0.05, method='bonferroni')[1]
    adj_ps_per_feat_booted = multipletests(ps_per_feat_booted, alpha=0.05, method='bonferroni')[1]
    for i in range(len(adj_ps_per_feat_random)):
        adj_ps_random.append(adj_ps_per_feat_random[i])
        adj_ps_booted.append(adj_ps_per_feat_booted[i])

df = pd.DataFrame()
df['feature'] = all_features
df['roi'] = all_rois
df['cor'] = all_cors
df['ps_random'] = adj_ps_random
df['ps_booted'] = adj_ps_booted
df['sign_random'] = np.array(adj_ps_random) < 0.025
df['sign_booted'] = np.array(adj_ps_booted) < 0.025
df['cis_lower'] = all_cis_lower
df['cis_upper'] = all_cis_upper

# Sort df on ROI order as in Lahner et al. (2023)
df['roi'] = pd.Categorical(df['roi'], ['V1v', 'V1d', 'V2v', 'V2d', 'V3d', 'V3v', 'hV4', 'EBA', 'FFA', 'OFA', 'STS', 'LOC', 'PPA', 'RSC', 'TOS', 'V3ab', 'IPS0', 'IPS1-2-3', '7AL', 'BA2', 'PFt', 'PFop'])
df['feature'] = pd.Categorical(df['feature'], ['objects', 'scenes', 'actions'])
df = df.sort_values(['roi', 'feature']).reset_index()

########################## Plot ##########################################

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

colors = {'objects': 'blue', 'scenes': 'red', 'actions': 'green'}

fig, ax = plt.subplots(dpi=400, figsize=(12,6))
sns.barplot(data=df, x="roi", y="cor", hue="feature", palette=colors)

all_patches = []
for i, p in enumerate(ax.patches):
    if i == len(df):
         break
    all_patches.append(p)

patch_count = 0
for i in range(df['feature'].nunique()):
    bar_xs = i + df['feature'].nunique() * np.arange(len(df['roi'].unique()))  # Calculate the x-coordinate for each bar
    for j in bar_xs:
        sign_random = df['sign_random'].iloc[j]
        sign_booted = df['sign_booted'].iloc[j]
        cor = df['cor'].iloc[j]
        ci_low = df['cis_lower'].iloc[j]
        ci_up = df['cis_upper'].iloc[j]
        p = all_patches[patch_count]

        if sign_random == True: 
            ax.annotate('*', (p.get_x() + p.get_width() / 2., 0), ha='center', va='center', xytext=(0, -10), textcoords='offset points', fontsize=8)
        if sign_booted == True: 
            ax.annotate('+', (p.get_x() + p.get_width() / 2., 0), ha='center', va='center', xytext=(0, -5), textcoords='offset points', fontsize=8, color='blue')
        ax.errorbar(x=(p.get_x() + p.get_width() / 2.), y=p.get_height(), yerr=np.array([cor - ci_low, ci_up - cor]).reshape(2, 1), ls='', lw=1, color='black')

        patch_count = patch_count + 1

ax.set_title(f'fmri-model correlation')
ax.set_xlabel('rois')
ax.set_ylabel('spearman')
ax.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
plt.xticks(rotation=45)
fig.tight_layout()

img_path = res_folder + f'random_vs_booted_plot.png'
plt.savefig(img_path)
plt.clf()

print('Done plotting')

