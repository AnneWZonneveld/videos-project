"""
Plot variance partioning results for model RDMs and fMRI RDM. 
Group level. 

Parameters
----------
data_split: str
    Train or test. 
alpha: float
    Significance threshold.
distance_type: str
    Whether to use EEG RDMs based on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
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
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--alpha', default=0.001, type=float)
parser.add_argument('--distance_type', default='pearson', type=str)
parser.add_argument('--model_metric', default='euclidean', type=str)

args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

####################### Load data #######################################
data_folder = f'/scratch/azonneveld/rsa/fusion/fmri-model/vp-results/{args.data_split}/model_{args.model_metric}/{args.distance_type}/'
res_folder = f'/scratch/azonneveld/rsa/fusion/fmri-model/plots/vp-results/{args.data_split}/model_{args.model_metric}/GA/{args.distance_type}/'
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)

cor_file  = data_folder + f'vp_results.pkl'
with open(cor_file, 'rb') as f: 
    cor_data = pickle.load(f)
cis_file = data_folder + f'vp_cis_results.pkl'
with open(cis_file, 'rb') as f: 
    cis_data = pickle.load(f)

models = ['u_o', 'u_s', 'u_a', 'os_shared', 'sa_shared', 'oa_shared', 'osa_shared', 'o_total', 's_total', 'a_total', 'all_parts']
rois = cor_data['results'].keys()

all_cors = []
all_ps = []
adj_ps = []
all_cis_lower = []
all_cis_upper = []
all_models = []
all_rois = []

for model in models:

    ps_per_model = []
    for roi in rois:
        all_cors.append(cor_data['results'][roi][0][model])
        all_ps.append(cor_data['results'][roi][1][model])
        ps_per_model.append(cor_data['results'][roi][1][model])
        all_cis_lower.append(cis_data['results'][roi][model][0])
        all_cis_upper.append(cis_data['results'][roi][model][1])
        all_rois.append(roi)
        
        if model == 'u_o':
            all_models.append('objects')
        elif model == 'u_s':
            all_models.append('scenes')
        elif model == 'u_a':
            all_models.append('actions')
        elif model == 'os_shared':
            all_models.append('o-s')
        elif model == 'oa_shared':
            all_models.append('o-a')
        elif model == 'sa_shared':
            all_models.append('s-a')
        elif model == 'osa_shared':
            all_models.append('o-s-a')
        elif model == 'o_total':
            all_models.append('o total')
        elif model == 's_total':
            all_models.append('s total')
        elif model == 'a_total':
            all_models.append('a total')
        elif model == 'all_parts':
            all_models.append('total')

    adj_ps_per_model = multipletests(ps_per_model, alpha=args.alpha, method='bonferroni')[1]
    for i in adj_ps_per_model:
        adj_ps.append(i)


df = pd.DataFrame()
df['model'] = all_models
df['roi'] = all_rois
df['cor'] = all_cors
df['ps'] = adj_ps
df['sign'] = np.array(adj_ps) < args.alpha #one sided
df['cis_lower'] = all_cis_lower
df['cis_upper'] = all_cis_upper


# Sort df on ROI order as in Lahner et al. (2023)
sorted_rois = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FFA', 'OFA', 'STS', 'LOC', 'PPA', 'RSC', 'TOS', 'V3ab', 'IPS0', 'IPS1-2-3', '7AL', 'BA2', 'PFt', 'PFop']

unique_models = ['objects', 'scenes', 'actions', 'total']
unique_df = df[df['model'].isin(unique_models)]
unique_df['roi'] = pd.Categorical(unique_df['roi'], sorted_rois)
unique_df['model'] = pd.Categorical(unique_df['model'], ['objects', 'scenes', 'actions', 'total'])
unique_df = unique_df.sort_values(['roi', 'model']).reset_index()

shared_models = ['o-s', 'o-a', 's-a', 'o-s-a']
shared_df = df[df['model'].isin(shared_models)]
shared_df['roi'] = pd.Categorical(shared_df['roi'], sorted_rois)
shared_df['model'] = pd.Categorical(shared_df['model'], ['o-a', 'o-s', 's-a', 'o-s-a'])
shared_df = shared_df.sort_values(['roi', 'model']).reset_index()

total_models = ['o total', 's total', 'a total', 'total']
total_df = df[df['model'].isin(total_models)]
total_df['roi'] = pd.Categorical(total_df['roi'], sorted_rois)
total_df['model'] = pd.Categorical(total_df['model'], ['o total', 's total', 'a total', 'total'])
total_df = total_df.sort_values(['roi', 'model']).reset_index()

########################## Plot ##########################################

sns.set_style('white')
sns.set_style("ticks")
sns.set_context('paper', 
                rc={'font.size': 14, 
                    'xtick.labelsize': 14, 
                    'ytick.labelsize':14, 
                    'axes.titlesize' : 13,
                    'figure.titleweight': 'bold', 
                    'axes.labelsize': 15, 
                    'legend.fontsize': 12, 
                    'font.family': 'Arial',
                    'axes.spines.right' : False,
                    'axes.spines.top' : False})


# ----------------- Unique plot
colors = {'objects': 'blue', 'scenes': 'red', 'actions': 'green', 'total': 'lightgray'}

fig, ax = plt.subplots(dpi=400, figsize=(12,6))
sns.barplot(data=unique_df, x="roi", y="cor", hue="model", palette=colors)

all_patches = []
for i, p in enumerate(ax.patches):
    if i == len(df):
         break
    all_patches.append(p)

patch_count = 0
for i in range(unique_df['model'].nunique()):
    bar_xs = i + unique_df['model'].nunique() * np.arange(len(df['roi'].unique()))  # Calculate the x-coordinate for each bar
    for j in bar_xs:
        sign = unique_df['sign'].iloc[j]
        cor = unique_df['cor'].iloc[j]
        ci_low = unique_df['cis_lower'].iloc[j]
        ci_up = unique_df['cis_upper'].iloc[j]
        p = all_patches[patch_count]

        if sign == True: 
            ax.annotate('*', (p.get_x() + p.get_width() / 2., 0), ha='center', va='center', xytext=(0, -5), textcoords='offset points', fontsize=8)

        patch_count = patch_count + 1

ax.set_title(f'fmri-model correlation')
ax.set_xlabel('rois')
ax.set_ylabel('Unique variance explained (%)')
ax.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
plt.xticks(rotation=45)
fig.tight_layout()
img_path = res_folder + f'unique_cor_plot.png'
plt.savefig(img_path)
plt.clf()

print('Done unique plotting')

for i in range(len(unique_df)):
    print(f'{unique_df.iloc[i]}')

# -------- Relative unique plot
relative_models = ['objects', 'scenes', 'actions']
relative_df = df[df['model'].isin(relative_models)]
relative_df['roi'] = pd.Categorical(relative_df['roi'], ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FFA', 'OFA', 'STS', 'LOC', 'PPA', 'RSC', 'TOS', 'V3ab', 'IPS0', 'IPS1-2-3', '7AL', 'BA2', 'PFt', 'PFop'])
relative_df['model'] = pd.Categorical(relative_df['model'], ['objects', 'scenes', 'actions'])
relative_df = relative_df.sort_values(['roi', 'model']).reset_index()

all_relative_values = []
for roi in sorted_rois:
    relative_values = (unique_df[(unique_df['roi'] == roi) & (unique_df['model'].isin(relative_models))]['cor'] / unique_df[(unique_df['roi'] == roi) & (unique_df['model'] == 'total')]['cor'].values[0]).values.tolist()
    for i in relative_values:
        all_relative_values.append(i * 100)

relative_df['relative cor'] = all_relative_values
    
colors = {'objects': 'blue', 'scenes': 'red', 'actions': 'green'}

fig, ax = plt.subplots(dpi=400, figsize=(12,6))
sns.barplot(data=relative_df, x="roi", y="relative cor", hue="model", palette=colors)

ax.set_title(f'fmri-model correlation')
ax.set_xlabel('rois')
ax.set_ylabel('Proportion variance explained (%)')
ax.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
plt.xticks(rotation=45)
fig.tight_layout()
img_path = res_folder + f'relative_unique_cor_plot.png'
plt.savefig(img_path)
plt.clf()

print('Done relative plotting')


# -------- Shared plot
colors = {'o-a': 'darkslateblue', 'o-s': 'darkcyan', 's-a': 'darkorange', 'o-s-a': 'palevioletred'}


fig, ax = plt.subplots(dpi=400, figsize=(12,6))
sns.barplot(data=shared_df, x="roi", y="cor", hue="model", palette=colors)

all_patches = []
for i, p in enumerate(ax.patches):
    if i == len(df):
         break
    all_patches.append(p)

patch_count = 0
for i in range(shared_df['model'].nunique()):
    bar_xs = i + shared_df['model'].nunique() * np.arange(len(shared_df['roi'].unique()))  # Calculate the x-coordinate for each bar
    for j in bar_xs:
        sign = shared_df['sign'].iloc[j]
        cor = shared_df['cor'].iloc[j]
        ci_low = shared_df['cis_lower'].iloc[j]
        ci_up = shared_df['cis_upper'].iloc[j]
        p = all_patches[patch_count]

        if sign == True: 
            ax.annotate('*', (p.get_x() + p.get_width() / 2., 0), ha='center', va='center', xytext=(0, -5), textcoords='offset points', fontsize=8)

        patch_count = patch_count + 1

ax.set_title(f'fmri-model correlation')
ax.set_xlabel('rois')
ax.set_ylabel('Shared variance explained (%)')
ax.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
plt.xticks(rotation=45)
fig.tight_layout()
img_path = res_folder + f'shared_cor_plot.png'
plt.savefig(img_path)
plt.clf()

print('Done shared plotting')
for i in range(len(shared_df)):
    print(f'{shared_df.iloc[i]}')

# -------- Relative shared plot
relative_s_models = ['o-a', 'o-s', 's-a', 'o-s-a']
relative_s_df = df[df['model'].isin(relative_s_models)]
relative_s_df['roi'] = pd.Categorical(relative_s_df['roi'], ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FFA', 'OFA', 'STS', 'LOC', 'PPA', 'RSC', 'TOS', 'V3ab', 'IPS0', 'IPS1-2-3', '7AL', 'BA2', 'PFt', 'PFop'])
relative_s_df['model'] = pd.Categorical(relative_s_df['model'], relative_s_models)
relative_s_df = relative_s_df.sort_values(['roi', 'model']).reset_index()

all_relative_values = []
for roi in sorted_rois:
    relative_values = (shared_df[(unique_df['roi'] == roi) & (shared_df['model'].isin(relative_s_models))]['cor'] / unique_df[(unique_df['roi'] == roi) & (unique_df['model'] == 'total')]['cor'].values[0]).values.tolist()
    for i in relative_values:
        all_relative_values.append(i * 100)

relative_s_df['relative cor'] = all_relative_values
    
colors = {'o-a': 'darkslateblue', 'o-s': 'darkcyan', 's-a': 'darkorange', 'o-s-a': 'palevioletred'}

fig, ax = plt.subplots(dpi=400, figsize=(12,6))
sns.barplot(data=relative_s_df, x="roi", y="relative cor", hue="model", palette=colors)

ax.set_title(f'fmri-model correlation')
ax.set_xlabel('rois')
ax.set_ylabel('Proportion variance explained (%)')
ax.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
plt.xticks(rotation=45)
fig.tight_layout()
img_path = res_folder + f'relative_shared_cor_plot.png'
plt.savefig(img_path)
plt.clf()

print('Done relative plotting')


# -------- Total plot
colors =  {'o total': 'royalblue', 's total': 'orangered', 'a total': 'limegreen', 'total' : 'lightgray'}

fig, ax = plt.subplots(dpi=400, figsize=(12,6))
sns.barplot(data=total_df, x="roi", y="cor", hue="model", palette=colors)

all_patches = []
for i, p in enumerate(ax.patches):
    if i == len(df):
         break
    all_patches.append(p)

patch_count = 0
for i in range(total_df['model'].nunique()):
    bar_xs = i + total_df['model'].nunique() * np.arange(len(total_df['roi'].unique()))  # Calculate the x-coordinate for each bar
    for j in bar_xs:
        sign = total_df['sign'].iloc[j]
        cor = total_df['cor'].iloc[j]
        ci_low = total_df['cis_lower'].iloc[j]
        ci_up = total_df['cis_upper'].iloc[j]
        p = all_patches[patch_count]

        if sign == True: 
            ax.annotate('*', (p.get_x() + p.get_width() / 2., 0), ha='center', va='center', xytext=(0, -5), textcoords='offset points', fontsize=8)

        patch_count = patch_count + 1

ax.set_title(f'fmri-model correlation')
ax.set_xlabel('rois')
ax.set_ylabel('Total variance explained (%)')
ax.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
plt.xticks(rotation=45)
fig.tight_layout()
img_path = res_folder + f'total_cor_plot.png'
plt.savefig(img_path)
plt.clf()

print('Done shared plotting')




