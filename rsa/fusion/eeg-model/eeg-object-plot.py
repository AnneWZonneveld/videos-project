"""
Plotting EEG-model fusion results comparing frequency based object model and average based object model. 

Code outdated.

Parameters
----------
sub: int
    Subject nr
alpha: float
    Significance threshold.
distance_type: str
    Whether to use EEG RDMs based on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
bin_width: int
    Bin width used for EEG smoothening

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
from statsmodels.stats.multitest import fdrcorrection
import argparse

# Take arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--disance_type', default='euclidean-cv', type=str)
parser.add_argument('--bin_width', default=0, type=float)

args = parser.parse_args()


####################### Load data #######################################
sub_format = format(args.sub, '02')
sub_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/standard/sub-{sub_format}/{args.distance_type}/bin_{args.bin_width}/' 
res_folder = f'/scratch/azonneveld/rsa/fusion/eeg-model/standard/plots/sub-{sub_format}/'

if not os.path.exists(res_folder) == True:
    os.mkdir(res_folder)

ob_types = ['avg', 'freq']
format_ob_types= ['', '_freq']

cors = {}
ps = {}
cis_low = {}
cis_up = {}

for ob_type in ob_types:
    cors[ob_type] = []
    ps[ob_type] = []
    cis_low[ob_type] = []
    cis_up[ob_type] = []  

for i in range(len(ob_types)):

    format_ob_type = format_ob_types[i]
    ob_type = ob_types[i]

    cor_path = sub_folder + f'cors_objects{format_ob_type}.pkl'
    ci_path = sub_folder + f'cis_objects{format_ob_type}.pkl'

    with open(cor_path, 'rb') as f: 
        cor_res = pickle.load(f)
    with open(ci_path, 'rb') as f: 
        ci_res = pickle.load(f)

    cor_values = cor_res['cor_values']
    cis_values =  ci_res['cis_values']
    for i in range(len(cor_values)):
        cors[ob_type].append(cor_values[i][0])
        ps[ob_type].append(cor_values[i][1])
        cis_low[ob_type].append(cis_values[i][0])
        cis_up[ob_type].append(cis_values[i][1])
    
    ps[ob_type] = fdrcorrection(ps[ob_type], alpha=args.alpha)[1]


######################### Plot ####################################
colours = ['b', 'r']
fig, ax = plt.subplots(dpi=300)
for i in range(len(ob_types)):
    ob_type = ob_types[i]
    colour = colours[i]

    stats_df = pd.DataFrame()
    stats_df['cors'] = cors[ob_type]
    stats_df['ps'] = np.array(ps[ob_type]) < args.alpha
    stats_df['lower_CI'] = cis_low[ob_type]
    stats_df['upper_CI'] = cis_up[ob_type]
    stats_df['times'] = cor_res['times']

    max_cor = np.max(stats_df['cors'])
    y_limit = 1.5 * max_cor
    format_ps = [y_limit if i == True else np.nan for i in stats_df['ps']]
    stats_df['format_ps'] = format_ps

    ax.plot(stats_df['times'], stats_df['cors'], label=ob_type, color=colour)
    ax.fill_between(stats_df['times'], stats_df['lower_CI'], stats_df['upper_CI'], color=colour, alpha=.1)
    ax.plot(stats_df['times'], stats_df['format_ps'], 'ro', color=colour)

ax.set_title('EEG-model correlation')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Spearman cor')
ax.legend()
fig.tight_layout()

img_path = res_folder + f'{sub_format}_objects_var_explained.png'
plt.savefig(img_path)
plt.clf()