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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, cut_tree, cophenet
import random
import scipy as sp
from scipy.spatial.distance import squareform, pdist

random.seed(1)

res_folder = '/scratch/azonneveld/clustering/plots' 

# Load metadata
with open(f'/scratch/azonneveld/rsa/md_global.pkl', 'rb') as f:
    md = pickle.load(f)


def ex_kmeans(fms, zet, var, n_clusters = 8, max_iter = 300, r_state = 0):
    fm = fms[zet][var]
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=r_state).fit(fm)

    return kmeans

def ex_hierarch(fms, zet, var, thres=None, n_clusters=None, linkage='ward'):
    fm = fms[zet][var]
    agglo = AgglomerativeClustering(distance_threshold=thres, n_clusters=n_clusters, compute_full_tree=True, linkage=linkage, compute_distances=True).fit(fm)

    return agglo


def elbow_plot(fms, zet, var, clus_range = range(5, 10), its = 5, c_type='kmean', linkage='ward', cb=True):

    if c_type == 'kmean':
        r_states = random.choices(range(1, 100), k=its)
        high_score = 0
        all_scores = []
        all_ks = []
        all_its = []
        all_av_cors = []

        for k in clus_range:

            print(f"performing kmeans {k} clusters")
        
            df = pd.DataFrame()

            for i in range(0, its):

                r_state = r_states[i]
                fit = ex_kmeans(fms, zet=zet, var=var, n_clusters=k, r_state=r_state)
                score = silhouette_score(fms[zet][var], fit.labels_, metric='euclidean')
                all_scores.append(score)
                all_ks.append(k)
                all_its.append(i)

                df[i] = fit.cluster_centers_.flatten()

                if score > high_score:
                    best_fit = fit
                    high_score = score
            
            cor_m = np.asarray(df.corr(method='spearman'))
            upper_triang = cor_m[np.triu_indices_from(cor_m, k = 1)]
            av_cor = np.mean(upper_triang)
            all_av_cors.append(av_cor)
    
        # Silhouette score
        elbow_df = pd.DataFrame()
        elbow_df['it'] = all_its
        elbow_df['k'] = all_ks
        elbow_df['score'] = all_scores

        fig, ax = plt.subplots(1,1)
        sns.lineplot(x='k', y='score', data=elbow_df)
        ax.set_ylabel('Silhouette score')
        ax.set_xlabel('N of clusters')
        ax.set_title(f'K means {zet} {var}, its = {its}')
        fig.tight_layout()

        img_path = res_folder + f'/kmeans_elbow_{zet}_{var}.png'
        plt.savefig(img_path)
        plt.clf()

        # Centroid variability between iterations
        fig, ax = plt.subplots(1,1)
        sns.lineplot(clus_range, all_av_cors)
        ax.set_ylabel('Spearman-r cor between iterations')
        ax.set_xlabel('N of clusters')
        ax.set_title(f'K means {zet} {var}, its = {its}')
        fig.tight_layout()

        img_path = res_folder + f'/kmeans_cor_{zet}_{var}.png'
        plt.savefig(img_path)
        plt.clf()

    elif c_type == 'hierarch':

        high_score = 0
        all_scores = []
        all_thres = []

        if cb == False:

            for thres in clus_range:

                print(f"performing clustering thres={thres}")
            
                df = pd.DataFrame()
                fit = ex_hierarch(fms=fms, zet=zet, var=var, thres=thres, linkage=linkage)
                score = silhouette_score(fms[zet][var], fit.labels_, metric='euclidean')
                all_scores.append(score)
                all_thres.append(thres)

                if score > high_score:
                    best_fit = fit
                    high_score = score
            
            elbow_df = pd.DataFrame()
            elbow_df['thres'] = all_thres
            elbow_df['score'] = all_scores

            fig, ax = plt.subplots(1,1)
            sns.lineplot(x='thres', y='score', data=elbow_df)
            ax.set_ylabel('Silhouette score')
            ax.set_xlabel('Distance threshold')
            ax.set_title(f'Hierach {zet} {var}')
            fig.tight_layout()
        else:

            for k in clus_range:

                print(f"performing clustering k={k}")
            
                df = pd.DataFrame()
                fit = ex_hierarch(fms=fms, zet=zet, var=var, n_clusters=k, linkage=linkage)
                score = silhouette_score(fms[zet][var], fit.labels_, metric='euclidean')
                all_scores.append(score)
                all_thres.append(k)

                if score > high_score:
                    best_fit = fit
                    high_score = score
            
            elbow_df = pd.DataFrame()
            elbow_df['k'] = all_thres
            elbow_df['score'] = all_scores

            fig, ax = plt.subplots(1,1)
            sns.lineplot(x='k', y='score', data=elbow_df)
            ax.set_ylabel('Silhouette score')
            ax.set_xlabel('K')
            ax.set_title(f'Hierach {zet} {var}')
            fig.tight_layout()


        img_path = res_folder + f'/hierarch_elbow_{zet}_{var}.png'
        plt.savefig(img_path)
        plt.clf()

    # Save best fit
    with open(f'/scratch/azonneveld/clustering/{c_type}_bf_{zet}_{var}_{linkage}.pkl', 'wb') as f:
        pickle.dump(best_fit, f)

def visual_inspect(zet, var, mds=True, count=True, k='bf', ctype='kmean'):

    # Load fit
    if k=='bf':
        with open(f'/scratch/azonneveld/clustering/{ctype}_bf_{zet}_{var}.pkl', 'rb') as f: 
            fit = pickle.load(f)
    else:
        with open(f'/scratch/azonneveld/clustering/{ctype}_k{k}_{zet}_{var}.pkl', 'rb') as f: 
            fit = pickle.load(f)

    # Only select relevant meta data
    md_select = md[md['set']==zet] 
    der_col = 'glb_' + var + '_lab'
    features = fms[zet][var]
    k = fit.n_clusters

    #MDS plot
    if mds == True:
        mds_model = MDS(n_components=2, random_state=0)
        mds_ft = mds_model.fit_transform(features)
        mds_df = pd.DataFrame(mds_ft, columns=['x', 'y'])
        mds_df['words'] = md_select[der_col].reset_index(drop=True)
        mds_df['clust'] = fit.labels_

        mds_plot = bp.figure(plot_width=500, plot_height=400, title=f"{ctype} {zet} {var} k={k} ",
        tools="pan,wheel_zoom,box_zoom,reset,hover",
        x_axis_type=None, y_axis_type=None, min_border=1)
        color_mapper = LinearColorMapper(palette='Turbo256', low=min(mds_df['clust']), high=max(mds_df['clust']))
        mds_plot.scatter(x='x', y='y', source=mds_df, color={'field': 'clust', 'transform': color_mapper})
        hover = mds_plot.select(dict(type=HoverTool))
        hover.tooltips={"word": "@words"}

        bp.output_file(filename=f"/scratch/azonneveld/clustering/plots/{ctype}_mds_{zet}_{var}_k{k}.html", title="mds-overview")
        bp.save(mds_plot)

    # Cluster count plot
    if count == True:
        # values, counts = np.unique(best_fit.labels_, return_counts=True)
        # df = pd.DataFrame()
        # df['count'] = counts
        # df['cluster'] = values
        # df = df.sort_values(by='count', axis=0, ascending=False).reset_index(drop=True)

        # fig, ax = plt.subplots(1,1)
        # sns.barplot(x='cluster', y='count', data=df, ax=ax, order=[*df.index])
        # ax.set_title(f'Kmeans {zet} {var}, k={k}', size=10)
        # ax.set_xticks([])

        # fig.tight_layout()
        # img_path = res_folder + f'/count_{zet}_{var}_k{k}.png'
        # plt.savefig(img_path)
        # plt.clf()

        df = pd.DataFrame(fit.labels_, columns=['cluster']).astype('category')

        fig, ax = plt.subplots(1,1)
        df['cluster'].value_counts().plot(kind="bar")
        ax.set_title(f'{ctype} {zet} {var}, k={k}', size=10)
        ax.set_xticks([])
        ax.set_ylabel('Count')
        ax.set_xlabel('Clusters')
        fig.tight_layout()
        img_path = res_folder + f'/{ctype}_count_{zet}_{var}_k{k}.png'
        plt.savefig(img_path)
        plt.clf()

def cluster_content(fit, zet, var):
    cluster_labels = fit.labels_.tolist()

    der_col = 'glb_' + var + '_lab'
    md_select = md[md['set']==zet][der_col].reset_index(drop=True).to_frame(name='label')
    md_select['cluster'] = cluster_labels

    clust_dict = {}
    for clust in np.unique(cluster_labels):
        data = md_select[md_select['cluster']==clust]
        label_unique, label_counts = np.unique(data['label'], return_counts=True)
        
        label_data = []
        for i in range(len(label_unique)):
            tup = (label_unique[i], label_counts[i])
            label_data.append(tup)

        clust_dict[clust] = label_data
    
    return clust_dict


def sample_ks(zet, var, clus_range, ctype='kmean'):

    if ctype == 'kmean':

        for k in clus_range:

            fit = ex_kmeans(fms, zet=zet, var=var, n_clusters=k)
            with open(f'/scratch/azonneveld/clustering/kmean_k{k}_{zet}_{var}.pkl', 'wb') as f:
                pickle.dump(fit, f)

            visual_inspect(zet=zet, var=var, k=k, ctype=ctype)
            #  clustr_contents(fit=fit, zet=zet, var=var)

    elif ctype == 'hierarch':
        
        for k in clus_range:

            fit = ex_hierarch(fms, zet=zet, var=var, n_clusters=k)
            with open(f'/scratch/azonneveld/clustering/hierarch_k{k}_{zet}_{var}.pkl', 'wb') as f:
                pickle.dump(fit, f)

            visual_inspect(zet=zet, var=var, k=k, ctype=ctype)


def plot_dendogram(zet, var, linkage, p=60, k='bf', **kwargs):

    if k == 'bf':
        # Load best fit
        with open(f'/scratch/azonneveld/clustering/hierarch_bf_{zet}_{var}.pkl', 'rb') as f: 
            fit = pickle.load(f)
    else:
        with open(f'/scratch/azonneveld/clustering/hierarch_k{k}_{zet}_{var}.pkl', 'rb') as f: 
            fit = pickle.load(f)


    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(fit.children_.shape[0])
    n_samples = len(fit.labels_)
    for i, merge in enumerate(fit.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [fit.children_, fit.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    fig, ax = plt.subplots(1,1, dpi=300)
    dendrogram(linkage_matrix,     
               truncate_mode='lastp',  
               p=p,  
               show_leaf_counts=True, 
               show_contracted=True,   
                **kwargs)

    try:
        ax.set_title(f'{zet} {var} {linkage}, thres={round(fit.distance_threshold, 3)}, k={k}')
    except: 
        ax.set_title(f'{zet} {var} {linkage}, k={k}')

    ax.set_ylabel('Distance')
    fig.tight_layout()
    img_path = res_folder + f'/dendogram_{zet}_{var}_k{k}.png'
    plt.savefig(img_path)
    plt.clf()
    
    return linkage_matrix


# ------ MAIN
with open(f'/scratch/azonneveld/rsa/fm_guse_glb.pkl', 'rb') as f: 
        fms = pickle.load(f)

# K-mean
# elbow_plot(fms, zet='train', var='actions', clus_range=range(2, 200), its=10, ctype='kmeans')
# elbow_plot(fms, zet='train', var='scenes', clus_range=range(2, 200), its=10, ctype='kmeans')
# elbow_plot(fms, zet='train', var='objects', clus_range=range(2, 200), its=3, ctype='kmeans')

# visual_inspect(zet='train', var='actions', k='bf')
# visual_inspect(zet='train', var='objects', k='bf')
# visual_inspect(zet='train', var='scenes', k='bf')

# Sample different k's
# sample_ks(zet='train', var='actions', clus_range=[20, 30, 40, 50, 60], ctype='kmean')
# sample_ks(zet='train', var='scenes', clus_range=[20, 30, 40, 50, 60], ctype='kmean')

# Hierarchical
# elbow_plot(fms, zet='train', var='scenes', clus_range=np.arange(0.1, 2, 0.01), c_type='hierarch', linkage='ward', cb=False)  #threshold
# elbow_plot(fms, zet='train', var='scenes', clus_range=np.arange(0.1, 2, 0.01), c_type='hierarch', linkage='average', cb=False) # threshold
# elbow_plot(fms, zet='train', var='scenes', clus_range=range(2, 200), c_type='hierarch', linkage='ward', cb=True) #cluster
# elbow_plot(fms, zet='train', var='actions', clus_range=range(2, 200), c_type='hierarch', linkage='ward', cb=True) #cluster

# Sample different k's
# sample_ks(zet='train', var='actions', clus_range=[20, 30, 40, 50, 60], ctype='hierarch')
# sample_ks(zet='train', var='scenes', clus_range=[20, 30, 40, 50, 60], ctype='hierarch')

# Plot dendogram
linkage_matrix = plot_dendogram(zet='train', var='actions', linkage='ward', k=40)
c, coph_dists = cophenet(linkage_matrix, pdist(fms['train']['actions']))

linkage_matrix = plot_dendogram(zet='train', var='scenes', linkage='ward', k=50)
c, coph_dists = cophenet(linkage_matrix, pdist(fms['train']['scenes']))


# # Manually cut tree
# num_clusters = 7
# clusters = cut_tree(linkage_matrix, n_clusters=num_clusters).flatten()