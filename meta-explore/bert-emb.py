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
from IPython import embed as shell
import scipy
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool, ColumnarDataSource, LinearColorMapper
from bokeh.layouts import row, column, layout
from sklearn.manifold import MDS
from transformers import BertTokenizer, BertModel

print("Downloading pretrained BERT")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

with open('/scratch/azonneveld/meta-explore/freq_data.pkl', 'rb') as f:
    freq_data = pickle.load(f)

# %% Get word vectors for labels
zets = ['train', 'test']
vars = ['objects', 'scenes', 'actions']

mds_plots = []
wv_dict = {}

# Irrelevant tokens are [CLS], [SEP], + and /
ir_tokens = [101, 102, 1009, 1013]

for zet in zets:
    temp_wv_dict = {}

    for var in vars:

        # Extract embeddings
        c_dict = freq_data[zet][var]
        labels = np.unique(c_dict['label'])
        n_labels = len(labels)

        print(f"Preprocessing embeddings {zet} {var}")
        wv = np.empty((n_labels, 768), dtype='object')
        for i in range(n_labels):
            label = labels[i]
            encoded_input = tokenizer(label, return_tensors='pt')
            tokens = encoded_input['input_ids']
            # check tokens: 
            # tokenizer.tokenize(label)

            output = model(**encoded_input)
            embedding = output.last_hidden_state

            # Check for (ir)relevant tokens
            idx_bool = np.isin(tokens, ir_tokens)[0]
            idx =  np.argwhere(idx_bool==False)

            # Concat relevant and average embeddings
            n_idx = idx.shape[0]
            concat_emb = np.empty((n_idx, 768))
            for j in range(n_idx):
                id = idx[j][0]
                concat_emb[j] = embedding[:, id, :].detach().numpy()

            av_emb = np.mean(concat_emb, axis=0)
            wv[i] = av_emb

            # Reduce to 300 for fair comparison fasttext?

        print(f"Performing MDS for {zet} {var}")
        mds_model = MDS(n_components=2, random_state=0)
        mds_ft = mds_model.fit_transform(wv)
        mds_df = pd.DataFrame(mds_ft, columns=['x', 'y'])
        mds_df['words'] = list(labels)
        mds_df['count'] = list(c_dict['count'])

        mds_plot = bp.figure(plot_width=500, plot_height=400, title=f"bert {zet} {var} labels",
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
with open(f'/scratch/azonneveld/meta-explore/bert_wv', 'wb') as f:
        pickle.dump(wv_dict, f)

# %% Save figure
complete_fig = layout([[mds_plots[0], mds_plots[1], mds_plots[2]],
        [mds_plots[3], mds_plots[4], mds_plots[5]]
])

bp.output_file(filename=f"/scratch/azonneveld/meta-explore/mds-overview-bert.html", title="mds-overview")
bp.save(complete_fig)

# # More evaluations
# vocab = ft.words # how to get the vocab?
# in_subset = ['home', 'love', 'understand', 'books', 'watch', 'town', 'hockey', 'friend', 'pay', 'need']
# out_subset = ['balance beam', 'zen garden', 'storage room', 'art school', 'ski slope', 'baseball player', 'hot pot', 'race car', 'playing music', 'coral reef'] # actually check 

# # Check --> true
# in_checks = []
# out_checks = []
# for i in range(len(out_subset)):
#     in_word = in_subset[i]
#     out_word = out_subset[i]
#     in_check = vocab.count(in_word)
#     out_check = vocab.count(out_word)
#     in_checks.append(in_check)
#     out_checks.append(out_check)
# print(f"in_checks: {in_checks}")
# print(f"out_checks: {out_checks}")

# in_neighbours = {}
# out_neighbours = {}
# for i in range(len(in_subset)):
#     in_word = in_subset[i]
#     in_neighbours[in_word] = ft.get_nearest_neighbors(in_word, k=5)
#     out_word = out_subset[i]
#     out_neighbours[out_word] = ft.get_nearest_neighbors(out_word, k=5)

# print(f"{model} in vocab neigh: ")
# print(f"{in_neighbours}")
# print(f"{model} out vocab neigh: ")
# print(f"{out_neighbours}")

# a1 = ft.get_analogies("berlin", "germany", "france") #paris
# a2 = ft.get_analogies("king", "prince", "queen") 
# a3 = ft.get_analogies("psx", "sony", "nintendo") # gamecube
# a4 = ft.get_analogies("ronaldo", "soccer", "tennis")
# print(a1)
# print(a2)
# print(a3)
# print(a4)
