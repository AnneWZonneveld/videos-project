import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool, ColumnarDataSource, LinearColorMapper
from bokeh.layouts import row, column, layout
import bokeh.palettes 
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random

random.seed(1)

res_folder = '/scratch/azonneveld/clustering/plots' 


def ex_kmeans(fms, zet, var, n_clusters = 8, max_iter = 300, r_state = 0):
    fm = fms[zet][var]
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=r_state).fit(fm)

    return kmeans

def elbow_plot(fms, zet, var, clus_range = range(5, 10), its = 5):

    high_score = 0

    fig, ax = plt.subplots(1,1)
    for i in range(0, its):

        r_state = random.randint(1, 100)
        print(f"Picking random start state {r_state}")

        scores = [] 

        for k in clus_range:
            print(f"performing kmeans {k} clusters")
            fit = ex_kmeans(fms, zet=zet, var=var, n_clusters=k, r_state=r_state)
            score = silhouette_score(fms[zet][var], fit.labels_, metric='euclidean')
            scores.append(score)

            if score > high_score:
                best_fit = fit
                high_score = score
        
        # plt.plot(clus_range, scores, label=i)
        sns.lineplot(clus_range, scores, ax=ax)

    ax.set_ylabel('Silhouette score')
    ax.set_xlabel('N of clusters')
    ax.set_title(f'K means {zet} {var}')
    fig.tight_layout()
    
    img_path = res_folder + f'/kmeans_elbow_{zet}_{var}.png'
    plt.savefig(img_path)
    plt.clf()

    # Save best fit
    with open(f'/scratch/azonneveld/clustering/kmean_bf_{zet}_{var}.pkl', 'wb') as f:
        pickle.dump(best_fit, f)

def visual_inspect(zet, var):

    # Load kmeans best fit
    with open(f'/scratch/azonneveld/clustering/kmean_bf_{zet}_{var}.pkl', 'rb') as f: 
        best_fit = pickle.load(f)

    # Load metadata
    with open(f'/scratch/azonneveld/rsa/md_global.pkl', 'rb') as f:
        md = pickle.load(f)

    # Only select train
    md = md[md['set']=='train'] 
    der_col = 'glb_' + var + '_lab'

    features = fms[zet][var]

    mds_model = MDS(n_components=2, random_state=0)
    mds_ft = mds_model.fit_transform(features)
    mds_df = pd.DataFrame(mds_ft, columns=['x', 'y'])
    mds_df['words'] = md[der_col].reset_index(drop=True)
    mds_df['clust'] = best_fit.labels_

    k = best_fit.n_clusters
    mds_plot = bp.figure(plot_width=500, plot_height=400, title=f"kmeans {zet} {var} k={k} ",
    tools="pan,wheel_zoom,box_zoom,reset,hover",
    x_axis_type=None, y_axis_type=None, min_border=1)
    color_mapper = LinearColorMapper(palette='Turbo256', low=min(mds_df['clust']), high=max(mds_df['clust']))
    mds_plot.scatter(x='x', y='y', source=mds_df, color={'field': 'clust', 'transform': color_mapper})
    hover = mds_plot.select(dict(type=HoverTool))
    hover.tooltips={"word": "@words"}

    bp.output_file(filename=f"/scratch/azonneveld/clustering/plots/mds_clusters_{zet}_{var}.html", title="mds-overview")
    bp.save(mds_plot)


# ------ MAIN
with open(f'/scratch/azonneveld/rsa/fm_guse_glb.pkl', 'rb') as f: 
        fms = pickle.load(f)

# elbow_plot(fms, zet='train', var='actions', clus_range=range(1, 200), its=5)
# elbow_plot(fms, zet='train', var='objects', clus_range=range(1, 200), its=5)
# elbow_plot(fms, zet='train', var='scenes', clus_range=range(1, 200), its=5)

visual_inspect(zet='train', var='actions')
# visual_inspect(zet='train', var='objects')
# visual_inspect(zet='train', var='scenes')
