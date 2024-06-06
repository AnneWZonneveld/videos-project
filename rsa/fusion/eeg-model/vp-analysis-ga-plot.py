"""
Plots results for variance partitioning on group level. To be performed after vp_analysis_ga_cor.py. 

Parameters
----------
data_split: str
    Train or test. 
alpha: float
    Significance threshold
zscore : int
	Whether to use z-scored EEG data [1] or not [0].
sfreq: int
    Sampling frequency of EEG data
distance_type: str
    Whether to use EEG RDMs based on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
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
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--sfreq', default=500, type=int)
args = parser.parse_args()


####################### Load data #######################################
sfreq_format = format(args.sfreq, '04')

sub_path = f'/scratch/azonneveld/rsa/fusion/eeg-model/{args.data_split}/vp-results/z_{args.zscore}/sfreq-{sfreq_format}/GA/vp_results.pkl' 
res_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/{args.data_split}/vp-results/plots/z_{args.zscore}/sfreq-{sfreq_format}/GA/'
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)

with open(sub_path, 'rb') as f: 
    vp_res = pickle.load(f)
vp_values = vp_res['results'][0]
p_values = vp_res['results'][1]
times = vp_res['times']

############ Temp: calc total #######################################

vp_values['total'] = vp_values['all_parts']
p_values['total'] = p_values['all_parts']
total_max = np.max(vp_values['total'])
total_mask = vp_values['total'] > 0.005 * total_max

######################### Plot: unique variance ####################################
print('Plotting unique variance explained')

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

features = ['objects', 'scenes', 'actions', 'total']
colours = ['b', 'r', 'g', 'lightgray']
max_value = np.max([vp_values['u_o'], vp_values['u_a'], vp_values['u_s'], vp_values['total']])
height_constant = 1.5

fig, axes = plt.subplots(dpi=300)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    if feature == 'objects':
        key = 'u_o'
    elif feature == 'scenes':
        key = 'u_s'
    elif feature == 'actions':
        key = 'u_a'
    elif feature == 'total':
        key = 'total'

    stats_df = pd.DataFrame()
    stats_df['values'] = vp_values[key]
    stats_df['times'] = times

    stats_df['ps'] =  np.array(multipletests(p_values[key], alpha=args.alpha, method='bonferroni')[1])< args.alpha
    y_limit = height_constant * max_value
    format_ps = [y_limit if i == True else np.nan for i in stats_df['ps']]
    stats_df['format_ps'] = format_ps
    sns.lineplot(data=stats_df, x='times', y='values', label=feature, color=colour)
    axes.plot(stats_df['times'], stats_df['format_ps'], 'ro', color=colour, markersize=2)

    for j in range(len(stats_df)):
        if stats_df['ps'].iloc[j] == True:
            onset_t = stats_df['times'].iloc[j]
            onset_cor = stats_df['values'].iloc[j]
            break

    try:
        print("########################################")
        print(f'Feature {feature} sign onset: {onset_t}')
        print(f'Cor : {onset_cor}')
        print("########################################")
    except:
        print('No sign timepoints')
    
    max_cor = np.max(vp_values[key])
    max_idx = np.where(vp_values[key] == max_cor)
    max_t = times[max_idx]

    print("########################################")
    print(f'Feature {feature} peak: {max_t}')
    print(f'Cor : {max_cor}')
    print("########################################")

    cor_35 = vp_values[key][35]
    t_35 = times[35]

    print("########################################")
    print(f'Feature {feature} second peak: {t_35}')
    print(f'Cor : {cor_35}')
    print("########################################")

    height_constant = height_constant - 0.1

axes.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
axes.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
axes.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes.set_title(f'EEG-model GA')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Unique variance exlained (%)')
axes.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

img_path = res_folder + f'GA_unique_r2_vp_bonferroni.png'
plt.savefig(img_path)
plt.clf()

######################### Plot: relative unique variance ####################################
print('Plotting relative unique variance explained')

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

features = ['objects', 'scenes', 'actions']
colours = ['b', 'r', 'g']

fig, axes = plt.subplots(dpi=300)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    if feature == 'objects':
        key = 'u_o'
    elif feature == 'scenes':
        key = 'u_s'
    elif feature == 'actions':
        key = 'u_a'

    stats_df = pd.DataFrame()
    stats_df['values'] = (np.array([0 if vp_values['total'][i] < 0.005 * total_max else vp_values[key][i] for i in range(len(vp_values['total']))])/ vp_values['total'])*100
    stats_df['times'] = times

    sns.lineplot(data=stats_df, x='times', y='values', label=feature, color=colour)

axes.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
axes.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
axes.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes.set_title(f'EEG-model GA')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Proportion variance exlained (%)')
axes.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

img_path = res_folder + f'GA_relative_unique_r2_vp_bonferroni.png'
plt.savefig(img_path)
plt.clf()

######################### Plot: cumalative unique variance ####################################
print('Plotting cum unique variance explained')

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

features = ['objects', 'scenes', 'actions']
colours = ['b', 'r', 'g']

fig, axes = plt.subplots(dpi=300)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    if feature == 'objects':
        key = 'u_o'
    elif feature == 'scenes':
        key = 'u_s'
    elif feature == 'actions':
        key = 'u_a'

    stats_df = pd.DataFrame()
    stats_df['values'] = np.cumsum(vp_values[key])
    stats_df['times'] = times

    sns.lineplot(data=stats_df, x='times', y='values', label=feature, color=colour)

axes.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
axes.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
axes.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes.set_title(f'EEG-model GA')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Cumalative variance exlained (%)')
axes.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

img_path = res_folder + f'GA_cum_unique_r2_vp_bonferroni.png'
plt.savefig(img_path)
plt.clf()


######################### Plot: total variance ####################################
# print('Plotting total variance explained')

# sns.set_style('white')
# sns.set_style("ticks")
# sns.set_context('paper', 
#                 rc={'font.size': 14, 
#                     'xtick.labelsize': 10, 
#                     'ytick.labelsize':10, 
#                     'axes.titlesize' : 13,
#                     'figure.titleweight': 'bold', 
#                     'axes.labelsize': 13, 
#                     'legend.fontsize': 8, 
#                     'font.family': 'Arial',
#                     'axes.spines.right' : False,
#                     'axes.spines.top' : False})

# features = ['objects', 'scenes', 'actions', 'all']
# colours = ['b', 'r', 'g', 'dargkgray']
# max_value = np.max([vp_values['o_total'], vp_values['s_total'], vp_values['a_total']])
# height_constant = 1.5

# fig, axes = plt.subplots(dpi=300)
# for i in range(len(features)):
#     feature = features[i]
#     colour = colours[i]

#     if feature == 'objects':
#         key = 'o_total'
#     elif feature == 'scenes':
#         key = 's_total'
#     elif feature == 'actions':
#         key = 'a_total'
#     elif feature == 'all':
#         key = 'all_parts'

#     stats_df = pd.DataFrame()
#     stats_df['values'] = vp_values[key]
#     stats_df['ps'] = np.array(fdrcorrection(p_values[key], alpha=args.alpha)[1]) < args.alpha
#     # stats_df['ps'] = np.array(p_values[key]) < alpha
#     stats_df['times'] = times
#     y_limit = height_constant * max_value
#     format_ps = [y_limit if i == True else np.nan for i in stats_df['ps']]
#     stats_df['format_ps'] = format_ps

#     sns.lineplot(data=stats_df, x='times', y='values', label=feature, color=colour)
#     axes.plot(stats_df['times'], stats_df['format_ps'], 'ro', color=colour, markersize=2)
    
#     height_constant = height_constant - 0.1

# axes.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
# axes.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
# axes.set_title(f'EEG-model {sub_format}')
# axes.set_xlabel('Time (s)')
# axes.set_ylabel('Variance exlained (%)')
# axes.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
# sns.despine(offset= 10, top=True, right=True)
# fig.tight_layout()

# img_path = res_folder + f'{sub_format}_total_r2_vp.png'
# plt.savefig(img_path)
# plt.clf()


######################### Plot: shared variance variance ####################################
print('Plotting shared variance')

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


features = ['o-a', 'o-s', 's-a', 'o-s-a']
colours = ['darkslateblue', 'darkcyan', 'darkorange', 'palevioletred']
max_value = np.max([vp_values['oa_shared'], vp_values['os_shared'], vp_values['sa_shared'], vp_values['osa_shared']])
height_constant = 1.5

fig, axes = plt.subplots(dpi=400)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    if feature == 'o-a':
        key = 'oa_shared'
    elif feature == 'o-s':
        key = 'os_shared'
    elif feature == 's-a':
        key = 'sa_shared'
    elif feature == 'o-s-a':
        key = 'osa_shared'

    stats_df = pd.DataFrame()
    stats_df['values'] = vp_values[key]
    stats_df['ps'] =  np.array(multipletests(p_values[key], alpha=args.alpha, method='bonferroni')[1])< args.alpha
    stats_df['times'] = times

    for j in range(len(stats_df)):
        if stats_df['ps'].iloc[j] == True:
            onset_t = stats_df['times'].iloc[j]
            onset_cor = stats_df['values'].iloc[j]
            break
    try:
        print("########################################")
        print(f'Feature {feature} sign onset: {onset_t}')
        print(f'Cor : {onset_cor}')
        print("########################################")
    except:
        print('No sign timepoints')
    
    max_cor = np.max(vp_values[key])
    max_idx = np.where(vp_values[key] == max_cor)
    max_t = times[max_idx]

    print("########################################")
    print(f'Feature {feature} peak: {max_t}')
    print(f'Cor : {max_cor}')
    print("########################################")

    cor_35 = vp_values[key][35]
    t_35 = times[35]

    print("########################################")
    print(f'Feature {feature} second peak: {t_35}')
    print(f'Cor : {cor_35}')
    print("########################################")


    y_limit = height_constant * max_value
    format_ps = [y_limit if i == True else np.nan for i in stats_df['ps']]
    stats_df['format_ps'] = format_ps

    sns.lineplot(data=stats_df, x='times', y='values', label=feature, color=colour)
    axes.plot(stats_df['times'], stats_df['format_ps'], 'ro', color=colour, markersize=2)
    
    height_constant = height_constant - 0.1

axes.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
axes.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
axes.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes.set_title(f'EEG-model GA')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Shared variance explained (%)')
axes.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

img_path = res_folder + f'GA_shared_r2_vp_bonferroni.png'
plt.savefig(img_path)
plt.clf()

print('Done plotting')

######################### Plot: relative shared variance ####################################
print('Plotting relative shared variance explained')

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

features = ['o-a', 'o-s', 's-a', 'o-s-a']
colours = ['darkslateblue', 'darkcyan', 'darkorange', 'palevioletred']

fig, axes = plt.subplots(dpi=400)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    if feature == 'o-a':
        key = 'oa_shared'
    elif feature == 'o-s':
        key = 'os_shared'
    elif feature == 's-a':
        key = 'sa_shared'
    elif feature == 'o-s-a':
        key = 'osa_shared'

    stats_df = pd.DataFrame()
    stats_df['values'] = (np.array([0 if vp_values['total'][i] < 0.005 * total_max else vp_values[key][i] for i in range(len(vp_values['total']))])/ vp_values['total'])*100
    stats_df['times'] = times

    sns.lineplot(data=stats_df, x='times', y='values', label=feature, color=colour)

axes.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
axes.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
axes.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes.set_title(f'EEG-model GA')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Proportion variance exlained (%)')
axes.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

img_path = res_folder + f'GA_relative_shared_r2_vp_bonferroni.png'
plt.savefig(img_path)
plt.clf()

######################### Plot: cumalative shared variance ####################################
print('Plotting relative shared variance explained')

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

features = ['o-a', 'o-s', 's-a', 'o-s-a']
colours = ['darkslateblue', 'darkcyan', 'darkorange', 'palevioletred']

fig, axes = plt.subplots(dpi=400)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    if feature == 'o-a':
        key = 'oa_shared'
    elif feature == 'o-s':
        key = 'os_shared'
    elif feature == 's-a':
        key = 'sa_shared'
    elif feature == 'o-s-a':
        key = 'osa_shared'

    stats_df = pd.DataFrame()
    stats_df['values'] =  np.cumsum(vp_values[key])
    stats_df['times'] = times

    sns.lineplot(data=stats_df, x='times', y='values', label=feature, color=colour)

axes.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
axes.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
axes.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes.set_title(f'EEG-model GA')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Cumalative variance exlained (%)')
axes.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

img_path = res_folder + f'GA_cum_shared_r2_vp_bonferroni.png'
plt.savefig(img_path)
plt.clf()