""" File to explore meta data BOLD moments"""

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

# Load json file to df
file_path = '/scratch/giffordale95/projects/eeg_videos/videos_metadata/annotations.json'
md = pd.read_json(file_path).transpose()

# Set image folder path
res_folder = '/scratch/azonneveld/meta-explore'

def load_freq_data():

    print('Loading data')

    # Inspect data structure
    col_names = md.columns.values.tolist()
    total_n_trials = md.shape[0]

    # Create count df for different label types train and test set
    zets = ['train', 'test']
    vars_oi = ['objects', 'scenes', 'actions']

    zets_dict = {}
    for zet in zets:
        temp_zet = md[md['set']==zet]
        n_trials = temp_zet.shape[0]
        vars_dict = {}

        for var in vars_oi:
            temp_df = temp_zet[var].to_numpy()
            all_labels = np.empty(5*n_trials, dtype=object)

            for i in range(5*n_trials):
                trial = i // 5
                j = i % 5 
                all_labels[i] = temp_df[trial][j]
            
            label_unique, label_counts = np.unique(all_labels, return_counts=True)
            label_unique = np.expand_dims(label_unique, axis = 1)
            label_counts = np.expand_dims(label_counts, axis = 1)

            count_df = pd.concat([pd.DataFrame(label_unique), pd.DataFrame(label_counts)], axis=1)
            count_df.columns = ['label', 'count']
            count_df = count_df.sort_values(by='count', axis=0, ascending=False)

            vars_dict[var] = count_df

        zets_dict[zet] = vars_dict

    # Save
    with open(res_folder+ '/freq_data.pkl', 'wb') as f:
        pickle.dump(zets_dict, f)

    return zets_dict

def freq_descript(freq_data):

    # Descriptives 
    zets = ['train', 'test']
    vars_oi = ['objects', 'scenes', 'actions']
    zets_max = {}

    for zet in zets:
        vars_dict = zets_dict[zet]
        labels_max = []

        for var in vars_oi:
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
    fig, ax = plt.subplots(2,3, dpi = 300, sharey='row')
    for j in range(len(zets)):
        zet = zets[j]
        vars_dict = zets_dict[zet]

        for i in range(len(vars_oi)):
            var = vars_oi[i]
            total_n = vars_dict[var].shape[0]

            sns.barplot(x='label', y='count', data=vars_dict[var], ax=ax[j, i])
            ax[j, i].set_title(f'{zet} {var}: {total_n}')
            ax[j, i].set_xticks([])
            ax[j, i].set_xlabel('Labels')

    fig.tight_layout()
    img_path = res_folder + '/label-hist.png'
    plt.savefig(img_path)
    plt.clf()

    # Plot top 15 label counts
    fig, ax = plt.subplots(2,3, dpi = 200, sharex='row')
    fig.set_figwidth(12)
    for j in range(len(zets)):
        zet = zets[j]
        vars_dict = zets_dict[zet]

        for i in range(len(vars_oi)):
            var = vars_oi[i]
            total_n = vars_dict[var].shape[0]

            sns.barplot(x='count', y='label', data=vars_dict[var].iloc[0:20], ax=ax[j, i])
            ax[j, i].set_xlabel('count')
            ax[j, i].set_ylabel('')
            ax[j,i].set_title(f'{zet} {var}: {total_n}')
            ax[j,i].tick_params(axis='y', labelsize=6)

    fig.tight_layout()
    img_path = res_folder + '/label-hist-top.png'
    plt.savefig(img_path)
    plt.clf()


# ------------------ Load freq data en plots
zets_dict = load_freq_data()
# freq_descript(zets_dict)

shell()
# ----------------- Make bokeh MDS plots
# %% Imports and fasttext 
import fasttext
import pickle
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool, ColumnarDataSource, LinearColorMapper
from bokeh.layouts import row, column, layout
from sklearn.manifold import MDS
import numpy as np
import pandas as pd

# Load word model 
print('Loading word model')
ft = fasttext.load_model('./downloads/cc.en.300.bin')

# %% Load freq dictionary 
with open('/scratch/azonneveld/meta-explore/freq_data.pkl', 'rb') as f:
    freq_data = pickle.load(f)

# # %% Get word vectors for labels
# vars_dict = freq_data['train']
# labels = np.unique(vars_dict['actions']['label'])
# n_labels = len(labels)

# wv = np.empty((n_labels, 300), dtype='object')
# for i in range(n_labels):
#     label = labels[i]
#     c_vect = ft.get_word_vector(label)
#     wv[i, :] = c_vect

# mds_model = MDS(n_components=2, random_state=0)
# mds_ft = mds_model.fit_transform(wv)
# mds_df = pd.DataFrame(mds_ft, columns=['x', 'y'])
# mds_df['words'] = list(labels)
# mds_df['count'] = list(vars_dict['actions']['count'])

# mds_plot = bp.figure(plot_width=500, plot_height=400, title="FastText Action labels",
# tools="pan,wheel_zoom,box_zoom,reset,hover",
# x_axis_type=None, y_axis_type=None, min_border=1)
# color_mapper = LinearColorMapper(palette='Plasma256', low=min(mds_df['count']), high=max(mds_df['count']))
# mds_plot.scatter(x='x', y='y', source=mds_df, color={'field': 'count', 'transform': color_mapper})
# hover = mds_plot.select(dict(type=HoverTool))
# hover.tooltips={"word": "@words",
#                 "count": "@count"}
# bp.output_file(filename="./meta-explore/mds-test.html", title="mds-test")
# bp.save(mds_plot)

# try:
#     bp.reset_output()
#     bp.output_notebook()
#     bp.show(mds_plot)
# except:
#     bp.output_notebook()
#     bp.show(mds_plot)


# %% Get word vectors for labels

zets = ['train', 'test']
vars = ['objects', 'scenes', 'actions']

mds_plots = []
wv_dict = {}

# Loop through sets and variables
for zet in zets:
    temp_wv_dict = {}

    for var in vars:

        # Extract embeddings
        c_dict = freq_data[zet][var]
        labels = np.unique(c_dict['label'])
        n_labels = len(labels)

        wv = np.empty((n_labels, 300), dtype='object')
        for i in range(n_labels):
            label = labels[i]
            c_vect = ft.get_word_vector(label)
            wv[i, :] = c_vect

        # Store embeddings
        temp_wv_dict[var] = wv

        # Perform MDS 
        print(f"Performing MDS for {zet} {var}")
        mds_model = MDS(n_components=2, random_state=0)
        mds_ft = mds_model.fit_transform(wv)
        mds_df = pd.DataFrame(mds_ft, columns=['x', 'y'])
        mds_df['words'] = list(labels)
        mds_df['count'] = list(c_dict['count'])

        mds_plot = bp.figure(plot_width=500, plot_height=400, title=f"FastText {zet} {var} labels",
        tools="pan,wheel_zoom,box_zoom,reset,hover",
        x_axis_type=None, y_axis_type=None, min_border=1)
        color_mapper = LinearColorMapper(palette='Plasma256', low=min(mds_df['count']), high=max(mds_df['count']))
        mds_plot.scatter(x='x', y='y', source=mds_df, color={'field': 'count', 'transform': color_mapper})
        hover = mds_plot.select(dict(type=HoverTool))
        hover.tooltips={"word": "@words",
                        "count": "@count"}
        
        mds_plots.append(mds_plot)

    wv_dict[zet] = temp_wv_dict

# %% Show figure
complete_fig = layout([[mds_plots[0], mds_plots[1], mds_plots[2]],
        [mds_plots[3], mds_plots[4], mds_plots[5]]
])

bp.output_file(filename="./meta-explore/mds-overview.html", title="mds-overview")
bp.save(complete_fig)

try:
    bp.reset_output()
    bp.output_notebook()
    bp.show(complete_fig)
except:
    bp.output_notebook()
    bp.show(complete_fig)

# %%
