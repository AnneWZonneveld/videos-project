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
from datasets import load_dataset, Dataset, DatasetDict
import faiss

descript = True

# Define functions
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

# Load data and model
with open('/scratch/azonneveld/meta-explore/freq_data.pkl', 'rb') as f:
    freq_data = pickle.load(f)

print("Downloading pretrained BERT")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

zets = ['train', 'test']
vars = ['objects', 'scenes', 'actions']

# MDS 
if os.path.exists("/scratch/azonneveld/meta-explore/mds-overview-bert.html") == False:
    # %% Get word vectors for labels

    mds_plots = []
    wv_dict = {}

    # # Irrelevant tokens are [CLS], [SEP], + and /
    # ir_tokens = [101, 102, 1009, 1013]

    dataset_wv_dict = {}
    all_labels = []
    all_embeddings = []

    for zet in zets:

        for var in vars:

            # Extract embeddings
            c_dict = freq_data[zet][var]
            labels = np.unique(c_dict['label'])
            n_labels = len(labels)

            print(f"Preprocessing embeddings {zet} {var}")
            wv = np.empty((n_labels, 768), dtype='object')
            
            for i in range(n_labels):
                label = labels[i]

                # # Mean pooling technique
                # encoded_input = tokenizer(label, return_tensors='pt')
                # tokens = encoded_input['input_ids']
                # # check tokens: 
                # # tokenizer.tokenize(label)

                # output = model(**encoded_input)
                # embedding = output.last_hidden_state

                # # Check for (ir)relevant tokens
                # idx_bool = np.isin(tokens, ir_tokens)[0]
                # idx =  np.argwhere(idx_bool==False)

                # # Concat relevant and average embeddings
                # n_idx = idx.shape[0]
                # concat_emb = np.empty((n_idx, 768))
                # for j in range(n_idx):
                #     id = idx[j][0]
                #     concat_emb[j] = embedding[:, id, :].detach().numpy()

                # av_emb = np.mean(concat_emb, axis=0)

                embedding = get_embeddings(label).detach().numpy()[0]
                wv[i] = embedding

                # Reduce to 300 for fair comparison fasttext?

                # Save all embeddings
                dataset_wv_dict[label] = embedding
                # all_labels.append(label)
                # all_embeddings.append(np.expand_dims(embedding, axis=0))

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

    # %% Save figure
    complete_fig = layout([[mds_plots[0], mds_plots[1], mds_plots[2]],
            [mds_plots[3], mds_plots[4], mds_plots[5]]
    ])

    bp.output_file(filename=f"/scratch/azonneveld/meta-explore/mds-overview-bert.html", title="mds-overview")
    bp.save(complete_fig)

    # Save all wvs
    with open(f'/scratch/azonneveld/meta-explore/bert_wv.pkl', 'wb') as f:
            pickle.dump(dataset_wv_dict, f)

# Descriptives
if descript == True:

    all_labels = []
    for zet in zets:
        for var in vars:
            c_dict = freq_data[zet][var]
            labels = np.unique(c_dict['label']).tolist()
            for label in labels:
                if not label in all_labels:
                    all_labels.append(label) 

    # Load hugging face dataset for FAISS
    ds_dict = {'labels': all_labels}
    ds = Dataset.from_dict(ds_dict)
    embeddings_ds = ds.map(
        lambda x: {"embeddings": get_embeddings(x['labels']).detach().numpy()[0]}
    )
    embeddings_ds.add_faiss_index(column='embeddings')

    # Get nearest neighbours for and out vocab
    in_subset = ['home', 'love', 'understand', 'books', 'watch', 'town', 'hockey', 'friend', 'pay', 'need']
    out_subset = ['balance beam', 'zen garden', 'storage room', 'art school', 'ski slope', 'baseball player', 'hot pot', 'race car', 'playing music', 'coral reef'] 

    in_neighbours = {}
    out_neighbours = {}
    for i in range(len(in_subset)):
        in_word = in_subset[i]
        # in_word_emb = np.expand_dims(np.asarray(embeddings_ds['embeddings'][embeddings_ds['labels']==in_word], dtype=np.float32), axis=0)

        in_word_emb = np.expand_dims(np.asarray(get_embeddings(in_word).detach().numpy()[0], dtype=np.float32), axis=0)

        scores, samples = embeddings_ds.get_nearest_examples("embeddings", in_word_emb, k=5)
        samples = samples['labels']
        in_res = (scores, samples)
        in_neighbours[in_word] = in_res

        out_word = out_subset[i]
        out_word_emb =  np.expand_dims(np.asarray(embeddings_ds['embeddings'][embeddings_ds['labels']==out_word], dtype=np.float32), axis=0)
        scores, samples = embeddings_ds.get_nearest_examples("embeddings", out_word_emb, k=5)
        samples = samples['labels']
        out_res = (scores, samples)               
        out_neighbours[out_word] = out_res

    print(f"bert in vocab neigh: ")
    print(f"{in_neighbours}")
    print(f"bert out vocab neigh: ")
    print(f"{out_neighbours}")
