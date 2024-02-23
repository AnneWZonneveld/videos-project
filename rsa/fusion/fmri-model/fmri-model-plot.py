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
parser.add_argument('--data_split', default='training', type=str)
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--distance_type', default='euclidean-cv', type=str)
parser.add_argument('--eval_method', default='spearman', type=str)
parser.add_argument('--model_metric', default='euclidean', type=str)

args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


####################### Load data #######################################
features = ['objects', 'scenes', 'actions']     

data_folder = f'/scratch/azonneveld/rsa/fusion/fmri-model/{args.data_split}/model_{args.model_metric}/{args.distance_type}/{args.eval_method}/'
res_folder = f'/scratch/azonneveld/rsa/fusion/fmri_model/plots/standard/{args.data_split}/{args.data_split}/model_{args.model_metric}/{args.distance_type}/{args.eval_method}/'
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)


all_cors = []
all_ps = []
all_cis_lower = []
all_cis_upper = []
all_features = []
all_rois = []

for feature in features:

    cor_file  = data_folder + f'cors_{feature}.pkl'
    with open(cor_file, 'rb') as f: 
        cor_data = pickle.load(f)
    cis_file = data_folder + f'cis_{feature}.pkl'
    with open(cis_file, 'rb') as f: 
        cis_data = pickle.load(f)

    rois = cor_data['results'].keys()
    for roi in rois:
        all_cors.append(cor_data['results'][roi][0])
        all_ps.append(cor_data['results'][roi][1])
        all_cis_lower.append(cis_data['results'][roi][0])
        all_cis_upper.append(cis_data['results'][roi][1])
        all_features.append(feature)
        all_rois.append(roi)

df = pd.DataFrame()
df['feature'] = all_features
df['roi'] = all_rois
df['cor'] = all_cors
df['ps'] = all_ps
# df['sign'] = np.array(all_ps) < args.alpha
# df['sign'] = np.array(all_ps) < 0.05
# df['sign'] = np.array(fdrcorrection(all_ps, alpha=args.alpha)[1])
df['sign'] = np.array(fdrcorrection(all_ps, alpha=0.05)[1]) < 0.05
df['cis_lower'] = all_cis_lower
df['cis_upper'] = all_cis_upper

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
# sns.barplot(data=df, x="roi", y="cor", hue="feature", palette=colors,
#             yerr= np.vstack((np.array(df['cor'] - df['cis_lower']), np.array(df['cis_upper'] - df['cor']))))
sns.barplot(data=df, x="roi", y="cor", hue="feature", palette=colors)

for i, p in enumerate(ax.patches):
    if i == 66:
         break
    sign = df['sign'].iloc[i]
    cor = df['cor'].iloc[i]
    ci_low = df['cis_lower'].iloc[i]
    ci_up = df['cis_upper'].iloc[i]
    if sign == True: 
        ax.annotate('*', (p.get_x() + p.get_width() / 2., 0), ha='center', va='center', xytext=(0, -10), textcoords='offset points', fontsize=8)
    ax.errorbar(x=(p.get_x() + p.get_width() / 2.), y=p.get_height(), yerr=np.array(cor - ci_low, ci_up - cor), ls='', lw=1, color='black')


ax.set_title(f'fmri-model correlation')
ax.set_xlabel('rois')
# ax.set_ylabel(f'{args.eval_method}')
ax.set_ylabel('spearman')
ax.legend(loc ='upper left', frameon=False, bbox_to_anchor=(1.04, 1))
sns.despine(offset= 10, top=True, right=True)
plt.xticks(rotation=45)
fig.tight_layout()

img_path = res_folder + f'cor_plot.png'
plt.savefig(img_path)
plt.clf()

print('Done plotting')





