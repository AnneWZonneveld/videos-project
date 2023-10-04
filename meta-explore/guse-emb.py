"""

- Extract BERT embeddings
- Perform MDS
- Make bokeh plots


"""
# %%
import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy
import tensorflow as tf
import tensorflow_hub as hub
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool, ColumnarDataSource, LinearColorMapper
from bokeh.layouts import row, column, layout
from sklearn.manifold import MDS
import re
from datasets import load_dataset, Dataset, DatasetDict

descript = False
perform_MDS = True

# Load freq data and model
with open('/scratch/azonneveld/meta-explore/freq_data.pkl', 'rb') as f:
    freq_data = pickle.load(f)

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)

zets = ['train', 'test']
vars = ['objects', 'scenes', 'actions']

# Functions
def embed(input):
  return model(input)


if perform_MDS == True: 
    # %% Get word vectors for labels
    mds_plots = []
    wv_dict = {}
    wv_dict_all = {}

    for zet in zets:

        zet_dict = {}

        for var in vars:

            var_dict = {}

            # Extract embeddings
            c_dict = freq_data[zet][var]
            labels = np.unique(c_dict['label'])
            n_labels = len(labels)

            print(f"Preprocessing embeddings {zet} {var}")
            wv = np.empty((n_labels, 512), dtype='object')
            
            for i in range(n_labels):
                label = labels[i]
                embedding = embed([label]).numpy()[0]
                wv[i] = embedding

                # Reduce to 300 for fair comparison fasttext?

                # Save all embeddings
                wv_dict_all[label] = embedding
                var_dict[label] = embedding
            
            zet_dict[var] = var_dict

            print(f"Performing MDS for {zet} {var}")
            mds_model = MDS(n_components=2, random_state=0)
            mds_ft = mds_model.fit_transform(wv)
            mds_df = pd.DataFrame(mds_ft, columns=['x', 'y'])
            mds_df['words'] = list(labels)
            mds_df['count'] = list(c_dict['count'])

            mds_plot = bp.figure(plot_width=500, plot_height=400, title=f"guse {zet} {var} labels",
            tools="pan,wheel_zoom,box_zoom,reset,hover",
            x_axis_type=None, y_axis_type=None, min_border=1)
            color_mapper = LinearColorMapper(palette='Plasma256', low=min(mds_df['count']), high=max(mds_df['count']))
            mds_plot.scatter(x='x', y='y', source=mds_df, color={'field': 'count', 'transform': color_mapper})
            hover = mds_plot.select(dict(type=HoverTool))
            hover.tooltips={"word": "@words",
                            "count": "@count"}

            mds_plots.append(mds_plot)
        
        wv_dict[zet] = zet_dict
            
    # %% Save figure
    complete_fig = layout([[mds_plots[0], mds_plots[1], mds_plots[2]],
            [mds_plots[3], mds_plots[4], mds_plots[5]]
    ])

    bp.output_file(filename=f"/scratch/azonneveld/meta-explore/mds-overview-guse.html", title="mds-overview")
    bp.save(complete_fig)

    # Save all wvs
    with open(f'/scratch/azonneveld/meta-explore/guse_wv_all.pkl', 'wb') as f:
            pickle.dump(wv_dict_all, f)
    with open(f'/scratch/azonneveld/meta-explore/guse_wv.pkl', 'wb') as f:
            pickle.dump(wv_dict, f)

# Descriptives
if descript == True:

    unique_vocab = []
    for zet in zets:
        for var in vars:
            c_dict = freq_data[zet][var]
            labels = np.unique(c_dict['label']).tolist()
            for label in labels:
                if not label in unique_vocab:
                    unique_vocab.append(label) 

    # Load hugging face dataset for FAISS
    ds_dict = {'labels': unique_vocab}
    ds = Dataset.from_dict(ds_dict)
    embeddings_ds = ds.map(
        lambda x: {"embeddings": embed([x['labels']]).numpy()[0]}
    )
    embeddings_ds.add_faiss_index(column='embeddings')
    
    # Get nearest neighbours for and out vocab
    # in_subset = ['home', 'love', 'understand', 'books', 'watch', 'town', 'hockey', 'friend', 'pay', 'need']
    # out_subset = ['balance beam', 'zen garden', 'storage room', 'art school', 'ski slope', 'baseball player', 'hot pot', 'race car', 'playing music', 'coral reef'] 

    in_subset = ['jellyfish', 'limousine', 'orange', 'sunglasses', 'shipwreck', 'bar', 'aquarium', 'walking', 'studying', 'smoking']
    out_subset = ['balance beam', 'zen garden', 'storage room', 'art school', 'ski slope', 'baseball player', 'hot pot', 'race car', 'playing+music', 'coral reef'] 

    in_neighbours = {}
    out_neighbours = {}
    for i in range(len(in_subset)):
        in_word = in_subset[i]
        in_word_emb = np.expand_dims(np.asarray(embed([in_word]).numpy()[0], dtype=np.float32), axis=0)       
        scores, samples = embeddings_ds.get_nearest_examples("embeddings", in_word_emb, k=5)
        samples = samples['labels']
        in_res = (scores, samples)
        in_neighbours[in_word] = in_res
        out_word = out_subset[i]
        out_word_emb = np.expand_dims(np.asarray(embed([out_word]).numpy()[0], dtype=np.float32), axis=0)                   
        scores, samples = embeddings_ds.get_nearest_examples("embeddings", out_word_emb, k=5)
        samples = samples['labels']
        out_res = (scores, samples)
        out_neighbours[out_word] = out_res

    print(f"{model} in vocab neigh: ")
    print(f"{in_neighbours}")
    print(f"{model} out vocab neigh: ")
    print(f"{out_neighbours}")
