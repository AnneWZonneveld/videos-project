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

# TEST
# test_text = "Whiskey flask"
# encoded_input = tokenizer(test_text, return_tensors='pt')
# tokenkizer.tokenize('word')
# output = model(**encoded_input)

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

        shell ()

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

        # Test plot
        bp.output_file(filename=f"/scratch/azonneveld/meta-explore/bert-test.html", title="bert-test") 
        bp.save(mds_plot)
        
    wv_dict[zet] = temp_wv_dict

# Save all wvs
with open(f'scratch/azonneveld/meta-explore/bert_wv', 'wb') as f:
        pickle.dump(wv_dict, f)

# %% Save figure
complete_fig = layout([[mds_plots[0], mds_plots[1], mds_plots[2]],
        [mds_plots[3], mds_plots[4], mds_plots[5]]
])

bp.output_file(filename=f"/scratch/azonneveld/meta-explore/mds-overview-bert.html", title="mds-overview")
bp.save(complete_fig)
