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
from statsmodels.stats.multitest import fdrcorrection

# Take arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--distance_type', default='euclidean-cv', type=str)
parser.add_argument('--bin_width', default=0, type=float)
parser.add_argument('--zscore', default=0, type=int)
parser.add_argument('--slide', default=1, type=int)
parser.add_argument('--cv_r2', default=0, type=int)
parser.add_argument('--ridge', default=0, type=int)
args = parser.parse_args()


####################### Load data #######################################
sub_format = format(args.sub, '02')
if args.bin_width == 0:
    bin_format = 0.0
else:
    bin_format = args.bin_width

sub_path = f'/scratch/azonneveld/rsa/fusion/eeg-model/reweighted/z_{args.zscore}/bin_{bin_format}/slide_{args.slide}/sub-{sub_format}/ridge_{args.ridge}/cv_{args.cv_r2}/vp_results.pkl' 
res_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/reweighted/plots/z_{args.zscore}/bin_{bin_format}/slide_{args.slide}/sub-{sub_format}/ridge_{args.ridge}/cv_{args.cv_r2}/'
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)

with open(sub_path, 'rb') as f: 
    vp_res = pickle.load(f)
vp_values = vp_res['results'][0]
p_values = vp_res['results'][1]
times = vp_res['times']

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

features = ['objects', 'scenes', 'actions']
colours = ['b', 'r', 'g']
max_value = np.max([vp_values['u_o'], vp_values['u_a'], vp_values['u_s']])
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

    stats_df = pd.DataFrame()
    stats_df['values'] = vp_values[key]
    stats_df['ps'] = np.array(fdrcorrection(p_values[key], alpha=args.alpha)[1]) < args.alpha
    # stats_df['ps'] = np.array(p_values[key]) < alpha
    stats_df['times'] = times
    y_limit = height_constant * max_value
    format_ps = [y_limit if i == True else np.nan for i in stats_df['ps']]
    stats_df['format_ps'] = format_ps

    sns.lineplot(data=stats_df, x='times', y='values', label=feature, color=colour)
    axes.plot(stats_df['times'], stats_df['format_ps'], 'ro', color=colour, markersize=2)
    
    height_constant = height_constant - 0.1

axes.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
axes.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
axes.set_title(f'EEG-model {sub_format}')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Unique variance exlained (%)')
axes.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

img_path = res_folder + f'{sub_format}_unique_r2_vp.png'
plt.savefig(img_path)
plt.clf()

######################### Plot: total variance ####################################
print('Plotting total variance explained')

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

total_o = np.array(vp_values['u_o']) + np.array(vp_values['oa_shared']) + np.array(vp_values['os_shared']) + np.array(vp_values['osa_shared'])
total_a = np.array(vp_values['u_a']) + np.array(vp_values['oa_shared']) + np.array(vp_values['sa_shared']) + np.array(vp_values['osa_shared'])
total_s = np.array(vp_values['u_s']) + np.array(vp_values['os_shared']) + np.array(vp_values['sa_shared']) + np.array(vp_values['osa_shared'])
total_all = np.array(vp_values['u_o']) + np.array(vp_values['u_a']) + np.array(vp_values['u_s']) + np.array(vp_values['os_shared']) + np.array(vp_values['oa_shared']) + np.array(vp_values['sa_shared']) + np.array(vp_values['osa_shared'])
total_data = {
    'objects' : total_o,
    'actions' : total_a,
    'scenes' : total_s,
    'all' : total_all
}

features = ['objects', 'scenes', 'actions', 'all']
colours = ['b', 'r', 'g', 'darkgrey']

fig, axes = plt.subplots(dpi=300)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    stats_df = pd.DataFrame()
    stats_df['values'] = total_data[feature]
    stats_df['times'] = times

    sns.lineplot(data=stats_df, x='times', y='values', label=feature, color=colour)
    

axes.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
axes.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
axes.set_title(f'EEG-model {sub_format}')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Variance exlained (%)')
axes.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

img_path = res_folder + f'{sub_format}_total_r2_vp.png'
plt.savefig(img_path)
plt.clf()


######################### Plot: shared variance variance ####################################
print('Plotting shared variance')

features = ['o & a', 'o & s', 's & a', 'all']
colours = ['darkslateblue', 'darkorchid', 'darkmagenta', 'darkorange']
max_value = np.max([vp_values['oa_shared'], vp_values['os_shared'], vp_values['sa_shared'], vp_values['osa_shared']])
height_constant = 1.5

fig, axes = plt.subplots(dpi=400)
for i in range(len(features)):
    feature = features[i]
    colour = colours[i]

    if feature == 'o & a':
        key = 'oa_shared'
    elif feature == 'o & s':
        key = 'os_shared'
    elif feature == 's & a':
        key = 'sa_shared'
    elif feature == 'all':
        key = 'osa_shared'

    stats_df = pd.DataFrame()
    stats_df['values'] = vp_values[key]
    stats_df['ps'] =  np.array(fdrcorrection(p_values[key], alpha=args.alpha)[1]) < args.alpha
    # stats_df['ps'] = np.array(p_values[key]) < alpha
    stats_df['times'] = times
    y_limit = height_constant * max_value
    format_ps = [y_limit if i == True else np.nan for i in stats_df['ps']]
    stats_df['format_ps'] = format_ps

    sns.lineplot(data=stats_df, x='times', y='values', label=feature, color=colour)
    axes.plot(stats_df['times'], stats_df['format_ps'], 'ro', color=colour, markersize=2)
    
    height_constant = height_constant - 0.1

axes.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
axes.axvline(x=3, color='gray', alpha=0.5, linestyle='--')
axes.set_title(f'EEG-model {sub_format}')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Shared variance exlained (%)')
axes.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

img_path = res_folder + f'{sub_format}_shared_r2_vp.png'
plt.savefig(img_path)
plt.clf()