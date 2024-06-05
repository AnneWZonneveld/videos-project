""" File to explore meta data BOLD moments

Should be performed after feature extraction (using rsa/model/model-rmds.py) if looking at distribution for derived labels
Gives overview of top-15 labels per visual event type (objects/scenes/actions) for train and test

"""

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
import scipy
import transformers
# from huggingface_hub import hf_hub_download
import fasttext
import fasttext.util
from sklearn.manifold import MDS
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook

# Set image folder path
res_folder = '/scratch/azonneveld/meta-explore'

# Meta data
file_path = '/scratch/azonneveld/downloads/annotations_humanScenesObjects.json'
md = pd.read_json(file_path).transpose()


def load_glob_md(ob_type = 'freq'):
    new_md = md.copy().reset_index(drop=True)
    vars = ['objects', 'scenes', 'actions']

    for var in vars:

        if var != 'objects':
            with open(f'/scratch/azonneveld/rsa/global_embs/{var}', 'rb') as f: 
                global_df = pickle.load(f)
        else:
            with open(f'/scratch/azonneveld/rsa/global_embs/{var}_{ob_type}', 'rb') as f: 
                global_df = pickle.load(f)

        new_md = pd.concat([new_md, global_df], axis=1)
    
    return new_md

def load_freq_data(lab_type='og'):
    """
    Lab_type: 'og' or 'derived'
    """

    print(f'Loading data {lab_type}')
    if lab_type == 'derived':
        md = load_glob_md(ob_type='freq')

    # Inspect data structure
    col_names = md.columns.values.tolist()
    total_n_trials = md.shape[0]

    # Create count df for different label types train and test set
    zets = ['train', 'test']
    vars_oi = ['objects', 'scenes', 'actions']
   
    zets_dict = {}
    for zet in zets:
        zet_df = md[md['set']==zet]
        n_trials = zet_df.shape[0]
        vars_dict = {}

        for var in vars_oi:

            if lab_type == 'derived':
                col_var =  'glb_' + var + '_lab'
                temp_df = zet_df[col_var].to_numpy()
            else:
                temp_df = zet_df[var].to_numpy()

            all_labels = []
            for i in range(n_trials):
                labels = temp_df[i]

                if lab_type == 'derived':
                    all_labels.append(labels)
                else:
                    for j in range(len(labels)):
                        label = labels[j]

                        if var in ['actions', 'scenes']:
                            all_labels.append(label)
                        else:
                            for lab in label:
                                if lab != '--':
                                    all_labels.append(lab)
            
            label_unique, label_counts = np.unique(np.array(all_labels), return_counts=True)
            label_unique = np.expand_dims(label_unique, axis = 1)
            label_counts = np.expand_dims(label_counts, axis = 1)

            count_df = pd.concat([pd.DataFrame(label_unique), pd.DataFrame(label_counts)], axis=1)
            count_df.columns = ['label', 'count']
            count_df = count_df.sort_values(by='count', axis=0, ascending=False)

            vars_dict[var] = count_df

        zets_dict[zet] = vars_dict

    # Save
    if lab_type == 'og':
        with open(res_folder+ '/freq_data.pkl', 'wb') as f:
            pickle.dump(zets_dict, f)
    if lab_type == 'derived':
        with open(res_folder+ f'/freq_data_der_freq.pkl', 'wb') as f:
            pickle.dump(zets_dict, f)

    return zets_dict

def freq_descript(freq_data, lab_type='og'):
    """
    Lab_type: 'og' or 'derived'
    """

    # Descriptives 
    zets = ['train', 'test']
    vars_oi = ['objects', 'scenes', 'actions']
    zets_max = {}

    for zet in zets:
        vars_dict = zets_dict[zet]
        labels_max = []

        for var in vars_oi:

            # if lab_type == 'derived':
            #     var =  'glb_' + var + '_lab'

            total_n = vars_dict[var].shape[0]
            max = vars_dict[var]['count'].iloc[0]
            labels_max.append(max)
            print(f'{zet} {var} unique labels: {total_n}')
            print(f'{zet} {var} top 10:')
            top_10 = vars_dict[var].iloc[0:10]
            print(f'{top_10}')

        zets_max[zet] = np.max(np.array(labels_max))

    print('Making plots')

    # Plot all label counts
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

    for j in range(len(zets)):
        zet = zets[j]
        vars_dict = zets_dict[zet]

        fig, axes = plt.subplots(1,3, dpi = 300, figsize=(34, 8), sharey='row')
        for i in range(len(vars_oi)):

            var = vars_oi[i]
            print(var)
            # if lab_type == 'derived':
            #     var =  'glb_' + var + '_lab'

            total_n = vars_dict[var].shape[0]
            data = vars_dict[var]
            x = data['label'].tolist()
            y = data['count'].tolist()

            # sns.barplot(x='label', y='count', data=data, ax=ax[i])
            sns.barplot(x=x, y=y, ax=axes[i])
            print(f'plot {i}')
            axes[i].set_title(f'{var}: {total_n}')
            axes[i].set_xticks([])
            # axes[i].set_xlabel('Label')

            if i == 0:
                axes[i].set_ylabel('Count')
        
        fig.supxlabel('Label')
        fig.suptitle(f'Distribution {zet} set')
        sns.despine(offset= 10, top=True, right=True)
        fig.tight_layout()
        img_path = res_folder + f'/{lab_type}-label-hist-{zet}.png'
        plt.savefig(img_path)
        plt.clf()


    # Plot top 15 label counts
    sns.set_palette('colorblind')    
    fig, ax = plt.subplots(2,3, dpi = 200, figsize=(20, 8), sharex='row')
    fig.set_figwidth(12)
    for j in range(len(zets)):
        zet = zets[j]
        vars_dict = zets_dict[zet]

        for i in range(len(vars_oi)):
            var = vars_oi[i]
            # if lab_type == 'derived':
            #     var =  'glb_' + var + '_lab'

            total_n = vars_dict[var].shape[0]

            sns.barplot(x='count', y='label', data=vars_dict[var].iloc[0:15], ax=ax[j, i])
            ax[j, i].set_xlabel('count')
            if i in [0, 4]:
                if j == 0:
                    ax[j, i].set_ylabel('Train')
                elif j == 1:
                    ax[j, i].set_ylabel('Test')
            else:
                ax[j, i].set_ylabel('')

            ax[j,i].set_title(f'{var}: {total_n}')
            # ax[j,i].tick_params(axis='y', labelsize=9)

    fig.suptitle(f'Top-15 labels')            
    sns.despine(offset= 10, top=True, right=True)
    fig.tight_layout()
    img_path = res_folder + f'/{lab_type}-label-hist-top.png'
    plt.savefig(img_path)
    plt.clf()


# ------------------ Load freq data en plots
zets_dict = load_freq_data(lab_type='og')
freq_descript(zets_dict, lab_type='og')

# zets_dict = load_freq_data(lab_type='derived')
# freq_descript(zets_dict, lab_type='derived')


