"""

- Extracts FastText embeddings
- Performs MDS
- Creates bokeh plots

"""

# %% Imports and fasttext 
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
import scipy
import fasttext
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool, ColumnarDataSource, LinearColorMapper
from bokeh.layouts import row, column, layout
from sklearn.manifold import MDS

#%%

# Pick 'ft' for default, 'cb_ft' for cbow, 'skip_ft' for skipgram
model = 'ft'

# Load word model: pretrained vs trained from scratch
print('Loading word model')
if model == 'ft':
    ft = fasttext.load_model('/scratch/azonneveld/downloads/cc.en.300.bin')
elif model == 'cb_ft':
    try:
        ft = fasttext.load_model('/scratch/azonneveld/downloads/cb.ft.300.bin')
    except:
        ft = fasttext.train_unsupervised('/scratch/azonneveld/downloads/fil9', model='cbow', dim=300)
        ft.save_model('/scratch/azonneveld/downloads/cb.ft.300.bin')
else:
    try:
        ft = fasttext.load_model('/scratch/azonneveld/downloads/skip.ft.300.bin')
    except:
        ft = fasttext.train_unsupervised('/scratch/azonneveld/downloads/fil9', model='skipgram', dim=300)
        ft.save_model('/scratch/azonneveld/downloads/skip.ft.300.bin')

# %% Load freq dictionary 
with open('/scratch/azonneveld/meta-explore/freq_data.pkl', 'rb') as f:
    freq_data = pickle.load(f)

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

        mds_plot = bp.figure(plot_width=500, plot_height=400, title=f"{model} {zet} {var} labels",
        tools="pan,wheel_zoom,box_zoom,reset,hover",
        x_axis_type=None, y_axis_type=None, min_border=1)
        color_mapper = LinearColorMapper(palette='Plasma256', low=min(mds_df['count']), high=max(mds_df['count']))
        mds_plot.scatter(x='x', y='y', source=mds_df, color={'field': 'count', 'transform': color_mapper})
        hover = mds_plot.select(dict(type=HoverTool))
        hover.tooltips={"word": "@words",
                        "count": "@count"}
        
        mds_plots.append(mds_plot)

    wv_dict[zet] = temp_wv_dict

# Save all wvs
with open(f'scratch/azonneveld/meta-explore/{model}_wv', 'wb') as f:
        pickle.dump(wv_dict, f)

# %% Save figure
complete_fig = layout([[mds_plots[0], mds_plots[1], mds_plots[2]],
        [mds_plots[3], mds_plots[4], mds_plots[5]]
])

bp.output_file(filename=f"/scratch/azonneveld/meta-explore/mds-overview-{model}.html", title="mds-overview")
bp.save(complete_fig)

# More evaluations
vocab = ft.words
in_subset = ['home', 'love', 'understand', 'books', 'watch', 'town', 'hockey', 'friend', 'pay', 'need']
out_subset = ['balance beam', 'zen garden', 'storage room', 'art school', 'ski slope', 'baseball player', 'hot pot', 'race car', 'playing music', 'coral reef'] # actually check 

# Check --> true
in_checks = []
out_checks = []
for i in range(len(out_subset)):
    in_word = in_subset[i]
    out_word = out_subset[i]
    in_check = vocab.count(in_word)
    out_check = vocab.count(out_word)
    in_checks.append(in_check)
    out_checks.append(out_check)

in_neighbours = {}
out_neighbours = {}
for i in range(len(in_subset)):
    in_word = in_subset[i]
    in_neighbours[in_word] = ft.get_nearest_neighbors(in_word, k=5)
    out_word = out_subset[i]
    out_neighbours[out_word] = ft.get_nearest_neighbors(out_word, k=5)

print(f"{model} in vocab neigh: ")
print(f"{in_neighbours}")
print(f"{model} out vocab neigh: ")
print(f"{out_neighbours}")

ft.get_analogies("berlin", "germany", "france") #paris
ft.get_analogies("king", "prince", "queen") 
ft.get_analogies("psx", "sony", "nintendo") # gamecube
ft.get_analogies("ronaldo", "soccer", "tennis")










#r_subset = np.random.choice(vocab, size=(100,), replace=False)




# shell()

# try:
#     bp.reset_output()
#     bp.output_notebook()
#     bp.show(complete_fig)
# except:
#     bp.output_notebook()
#     bp.show(complete_fig)
