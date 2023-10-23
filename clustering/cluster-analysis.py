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
        all_fits = []

        for k in clus_range:

            print(f"performing kmeans {k} clusters")
        
            df = pd.DataFrame()

            for i in range(0, its):

                r_state = r_states[i]
                fit = ex_kmeans(fms, zet=zet, var=var, n_clusters=k, r_state=r_state)
                all_fits.append(fit)
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
        all_fits = []
        
        if cb == False:

            all_thres = []
            all_ks = []

            for thres in clus_range:

                print(f"performing clustering thres={thres}")
            
                df = pd.DataFrame()
                fit = ex_hierarch(fms=fms, zet=zet, var=var, thres=thres, linkage=linkage)
                all_fits.append(fit)
                score = silhouette_score(fms[zet][var], fit.labels_, metric='euclidean')
                all_scores.append(score)
                all_thres.append(thres)
                all_ks.append(fit.n_clusters_)

                if score > high_score:
                    best_fit = fit
                    high_score = score
            
            elbow_df = pd.DataFrame()
            elbow_df['thres'] = all_thres
            elbow_df['score'] = all_scores
            elbow_df['k'] = all_ks


            fig, ax1 = plt.subplots()
            sns.lineplot(x='thres', y='score', data=elbow_df, color='blue', ax=ax1)
            ax1.set_ylabel('Silhouette score')
            ax1.set_xlabel('Distance threshold')
            ax1.set_title(f'Hierach {zet} {var} {linkage}')
            ax2 = ax1.twinx()
            sns.lineplot(x='thres', y='k', data=elbow_df, color='red', ax=ax2)
            ax2.set_ylabel('K')
            if var == 'actions':
                line_val = 179
            elif var == 'scenes':
                line_val = 189
            plt.axhline(y=line_val, color='black', label=f'n of unique labels ({line_val})')
            best_thres = best_fit.distance_threshold
            plt.axvline(x=best_thres, color='orange', label=f'optimal threshold ({round(best_thres, 2)})')

            fig.tight_layout()
            img_path = res_folder + f'/hierarch_elbow_{zet}_{var}_{linkage}_thres.png'

        else:

            all_ks = []

            for k in clus_range:

                print(f"performing clustering k={k}")
            
                df = pd.DataFrame()
                fit = ex_hierarch(fms=fms, zet=zet, var=var, n_clusters=k, linkage=linkage)
                all_fits.append(fit)
                score = silhouette_score(fms[zet][var], fit.labels_, metric='euclidean')
                all_scores.append(score)
                all_ks.append(k)

                if score > high_score:
                    best_fit = fit
                    high_score = score
            
            elbow_df = pd.DataFrame()
            elbow_df['k'] = all_ks
            elbow_df['score'] = all_scores

            fig, ax = plt.subplots(1,1)
            sns.lineplot(x='k', y='score', data=elbow_df)
            ax.set_ylabel('Silhouette score')
            ax.set_xlabel('K')
            ax.set_title(f'Hierach {zet} {var}')
            fig.tight_layout()
            img_path = res_folder + f'/hierarch_elbow_{zet}_{var}_{linkage}.png'


        plt.savefig(img_path)
        plt.clf()

    # Save best fit
    if c_type == 'hierarch':
        linkage = "_" + linkage
    with open(f'/scratch/azonneveld/clustering/{c_type}_bf_{zet}_{var}{linkage}.pkl', 'wb') as f:
        pickle.dump(best_fit, f)
    
    # Save all fits
    with open(f'/scratch/azonneveld/clustering/{c_type}_all_{zet}_{var}{linkage}.pkl', 'wb') as f:
        pickle.dump(all_fits, f)

def fits_variation_plot(zet, var, c_type, linkage, var_type='median'):

    # Load all fits 
    if c_type == 'hierarch':
        linkage = "_" + linkage
    with open(f'/scratch/azonneveld/clustering/{c_type}_all_{zet}_{var}{linkage}.pkl', 'rb') as f:
        all_fits = pickle.load(f)
    with open(f'/scratch/azonneveld/clustering/{c_type}_bf_{zet}_{var}{linkage}.pkl', 'rb') as f:
        best_fit = pickle.load(f)
    
    
    all_means = []
    all_sds = []
    all_thres = []
    all_q3 = []
    all_q2 = []
    all_q1 = []
    for fit in all_fits:
        labels =  fit.labels_
        values, counts = np.unique(labels, return_counts=True)
        mean = np.mean(counts)
        sd = np.std(counts)
        thres = fit.distance_threshold
        q3, q2, q1 = np.percentile(counts, [75, 50, 25])
        all_q3.append(q3)
        all_q2.append(q2)
        all_q1.append(q1)
        all_means.append(mean)
        all_sds.append(sd)
        all_thres.append(thres)

    size_df = pd.DataFrame()
    size_df['mean'] = all_means
    size_df['q2'] = all_q2
    size_df['sd'] = all_sds
    size_df['thres'] = all_thres
    upper = np.asarray(all_means) + np.asarray(all_sds)
    lower = np.asarray(all_means) -  np.asarray(all_sds)
    all_thres = np.asarray(all_thres)
    all_q3 = np.asarray(all_q3)
    all_q1 = np.asarray(all_q1)

    best_thres = best_fit.distance_threshold

    if var_type == 'median':
        fig, ax1 = plt.subplots(dpi=300)
        sns.lineplot(x='thres', y='q2', data=size_df, color='blue', ax=ax1)
        ax1.set_ylabel('Cluster size')
        ax1.set_xlabel('Distance threshold')
        ax1.set_title(f'IQR size {c_type} {zet} {var} {linkage}')
        plt.fill_between(all_thres, all_q1, all_q3, alpha=.3)
        plt.axvline(x=best_thres, color='orange')

        fig.tight_layout()
        img_path = res_folder + f'/{c_type}_size_{zet}_{var}_{linkage}_thres.png'
        plt.savefig(img_path)
        plt.clf()

    elif var_type == 'mean':
        fig, ax1 = plt.subplots(dpi=300)
        sns.lineplot(x='thres', y='mean', data=size_df, color='blue', ax=ax1)
        ax1.set_ylabel('Cluster size')
        ax1.set_xlabel('Distance threshold')
        ax1.set_title(f'Mean size {c_type} {zet} {var} {linkage}')
        plt.fill_between(all_thres, lower, upper, alpha=.3)
        plt.axvline(x=best_thres, color='orange')

        fig.tight_layout()
        img_path = res_folder + f'/{c_type}_size_{zet}_{var}_{linkage}_thres.png'
        plt.savefig(img_path)
        plt.clf()


def visual_inspect(zet, var, mds=True, count=True, k='bf', ctype='kmean', linkage=''):

    # Load fit
    if ctype == 'hierarch':
        linkage = "_" + linkage

    if k=='bf':
        with open(f'/scratch/azonneveld/clustering/{ctype}_bf_{zet}_{var}{linkage}.pkl', 'rb') as f: 
            fit = pickle.load(f)
    else:
        with open(f'/scratch/azonneveld/clustering/{ctype}_k{k}_{zet}_{var}{linkage}.pkl', 'rb') as f: 
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

        mds_plot = bp.figure(plot_width=500, plot_height=400, title=f"{ctype}{linkage} {zet} {var} k={k} ",
        tools="pan,wheel_zoom,box_zoom,reset,hover",
        x_axis_type=None, y_axis_type=None, min_border=1)
        color_mapper = LinearColorMapper(palette='Turbo256', low=min(mds_df['clust']), high=max(mds_df['clust']))
        mds_plot.scatter(x='x', y='y', source=mds_df, color={'field': 'clust', 'transform': color_mapper})
        hover = mds_plot.select(dict(type=HoverTool))
        hover.tooltips={"word": "@words",
                        "clust": "@clust"}

        bp.output_file(filename=f"/scratch/azonneveld/clustering/plots/{ctype}_mds_{zet}_{var}_k{k}_{linkage}.html", title="mds-overview")
        bp.save(mds_plot)

    # Cluster count plot
    if count == True:
        df = pd.DataFrame(fit.labels_, columns=['cluster']).astype('category')

        fig, ax = plt.subplots(1,1)
        df['cluster'].value_counts().plot(kind="bar")
        ax.set_title(f'{ctype}{linkage} {zet} {var}, k={k}', size=10)
        ax.set_xticks([])
        ax.set_ylabel('Count')
        ax.set_xlabel('Clusters')
        fig.tight_layout()
        img_path = res_folder + f'/{ctype}_count_{zet}_{var}_k{k}_{linkage}.png'
        plt.savefig(img_path)
        plt.clf()


def cluster_content(zet, var, linkage, k=2):
      
    fit = ex_hierarch(fms=fms, zet=zet, var=var, n_clusters=k, linkage='ward')              
    cluster_labels = fit.labels_.tolist()
    n_clusters = len(np.unique(cluster_labels))

    der_col = 'glb_' + var + '_lab'
    md_select = md[md['set']==zet][der_col].reset_index(drop=True).to_frame(name='label')
    md_select['cluster'] = cluster_labels

    count_df = md_select.groupby(['cluster','label']).size().to_frame(name='count').reset_index()
    count_df = count_df.sort_values(by=['cluster', 'count'],
                    )

    ncols = 5
    nrows = n_clusters // ncols + (n_clusters % ncols > 0)

    fig = plt.figure(dpi=300, figsize=(22, 16))
    for i in range(n_clusters):
        cluster_df = count_df[count_df['cluster']== i]
        cluster_size = cluster_df['count'].sum() 
        ax  = plt.subplot(nrows, ncols, i+1)       
        sns.barplot(data = cluster_df, x='label', y='count', ax = ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=6)
        ax.set_xlabel('')
        ax.set_title(f'cluster {i} ({cluster_size})', size=8)
    fig.suptitle(f'{linkage}, k={n_clusters}')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    img_path = res_folder + f'/cluster_content/{linkage}_k{k}.png'
    plt.savefig(img_path)
    plt.clf()

    # TEst
    fit = ex_hierarch(fms=fms, zet=zet, var=var, n_clusters=k, linkage='ward') 
    with open(f'/scratch/azonneveld/meta-explore/guse_wv_all.pkl', 'rb') as f: 
        wvs = pickle.load(f)
    
    temp_fm = np.zeros((len(wvs.keys()), 512))
    for i 
    





def clus_size_dist():
    pass


def sample_ks(zet, var, clus_range, ctype='kmean', linkage=''):


    for k in clus_range:

        if ctype == 'kmean':
            fit = ex_kmeans(fms, zet=zet, var=var, n_clusters=k)
            link_label = linkage
        elif ctype == 'hierarch':  
            fit = ex_hierarch(fms, zet=zet, var=var, n_clusters=k, linkage=linkage)
            link_label  = '_' + linkage

        with open(f'/scratch/azonneveld/clustering/{ctype}_k{k}_{zet}_{var}{link_label}.pkl', 'wb') as f:
            pickle.dump(fit, f)
        
        visual_inspect(zet=zet, var=var, k=k, ctype=ctype, linkage=linkage)


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

    c, coph_dists = cophenet(linkage_matrix, pdist(fms[zet][var]))


    # Plot the corresponding dendrogram
    fig, ax = plt.subplots(1,1, dpi=300)
    dendrogram(linkage_matrix,     
               truncate_mode='lastp',  
               p=p,  
               show_leaf_counts=True, 
               show_contracted=True,   
                **kwargs)
    ax.set_title(f'{zet} {var} {linkage}, coph_c={c}')
    ax.set_ylabel('Distance')
    fig.tight_layout()
    img_path = res_folder + f'/dendogram_{zet}_{var}_{linkage}.png'
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

# visual_inspect(zet='train', var='actions', k=40, ctype='kmean')
# visual_inspect(zet='train', var='actions', k=40, ctype='hierarch')

# Sample different k's
# sample_ks(zet='train', var='actions', clus_range=[20, 30, 40, 50, 60], ctype='kmean')
# sample_ks(zet='train', var='scenes', clus_range=[20, 30, 40, 50, 60], ctype='kmean')

# Hierarchical
# elbow_plot(fms, zet='train', var='scenes', clus_range=np.arange(0.1, 2, 0.01), c_type='hierarch', linkage='ward', cb=False)  #threshold
# elbow_plot(fms, zet='train', var='scenes', clus_range=np.arange(0.1, 1.08, 0.01), c_type='hierarch', linkage='average', cb=False) # threshold
# elbow_plot(fms, zet='train', var='scenes', clus_range=np.arange(0.1, 0.75, 0.01), c_type='hierarch', linkage='single', cb=False)  #threshold
# elbow_plot(fms, zet='train', var='scenes', clus_range=np.arange(0.1, 0.8, 0.01), c_type='hierarch', linkage='complete', cb=False) # threshold

# elbow_plot(fms, zet='train', var='actions', clus_range=np.arange(0.1, 2, 0.01), c_type='hierarch', linkage='ward', cb=False)  #threshold
# elbow_plot(fms, zet='train', var='actions', clus_range=np.arange(0.1, 1.08, 0.01), c_type='hierarch', linkage='average', cb=False) # threshold
# elbow_plot(fms, zet='train', var='actions', clus_range=np.arange(0.1, 0.75, 0.01), c_type='hierarch', linkage='single', cb=False)  #threshold
# elbow_plot(fms, zet='train', var='actions', clus_range=np.arange(0.1, 0.8, 0.01), c_type='hierarch', linkage='complete', cb=False) # threshold

# Asses cluster size variation
# fits_variation_plot(zet='train', var='scenes', c_type='hierarch', linkage='ward')
# fits_variation_plot(zet='train', var='scenes', c_type='hierarch', linkage='average')
# fits_variation_plot(zet='train', var='scenes', c_type='hierarch', linkage='single')
# fits_variation_plot(zet='train', var='scenes', c_type='hierarch', linkage='complete')
# fits_variation_plot(zet='train', var='actions', c_type='hierarch', linkage='ward')
# fits_variation_plot(zet='train', var='actions', c_type='hierarch', linkage='average')
# fits_variation_plot(zet='train', var='actions', c_type='hierarch', linkage='single')
# fits_variation_plot(zet='train', var='actions', c_type='hierarch', linkage='complete')

# Sample different k's
# sample_ks(zet='train', var='actions', clus_range=[20, 30, 40, 50, 60], ctype='hierarch')
# sample_ks(zet='train', var='scenes', clus_range=[20, 30, 40, 50, 60], ctype='hierarch')

# Plot dendogram
# linkage_matrix = plot_dendogram(zet='train', var='actions', linkage='ward', k=40)

# Asses cluster content for different points hierarchy
ks = 20
for k in range(2, ks):
    print(f'Assesing content k={k}')
    cluster_content(zet='train', var='actions', linkage='ward', k=k)