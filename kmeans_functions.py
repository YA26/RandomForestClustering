import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from kneed import KneeLocator
from os.path import join
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
"""
##############################################
########## USEFUL FUNCTIONS FOR K-MEANS#######
##############################################
"""
def compute_sse(coordinates, n_clusters, kmeans_kwargs):
    models = defaultdict()
    sse = []
    for k in range(1, n_clusters+1):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(coordinates)
        sse.append(kmeans.inertia_)
        models[k] = kmeans
    return {"sse":sse, 
            "models":models}
  
      
def compute_silhouette(coordinates, n_clusters, kmeans_kwargs): 
    models = defaultdict()
    silhouette_coefficients = []
    for k in range(2, n_clusters+2):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(coordinates)
        score = silhouette_score(coordinates, kmeans.labels_)
        silhouette_coefficients.append(score)
        models[k] = kmeans
    return {"silhouette_scores":silhouette_coefficients, 
           "models":models}
 
    
def plot_scores(scores, title, n_clusters, type_="sse", path_to_save=None): 
    plt.style.use("fivethirtyeight")
    if type_.strip()=="sse":
        plt.plot(range(1, n_clusters+1), scores)
        plt.xticks(range(1, n_clusters+1))
    else:
        plt.plot(range(2, n_clusters+2), scores)
        plt.xticks(range(2, n_clusters+2))
    plt.xlabel("Number of Clusters")
    plt.ylabel(title)
    plt.show()
    if path_to_save:
        plt.savefig(join(path_to_save, f"{title}.jpg"))
    plt.close()


def get_optimal_n_clusters(n_clusters, 
                           scores, 
                           type_="sse", 
                           curve="convex", 
                           direction="decreasing"):
    range_=None
    if type_.strip()=="sse":
        range_ = range(1, n_clusters+1)
    else:
        range_ = range(2, n_clusters+2)
    kl = KneeLocator(range_, 
                     scores, 
                     curve=curve, 
                     direction=direction)
    return kl.elbow


def plot_clusters(model, 
                  coordinates, 
                  predictions, 
                  colors, 
                  figsize, 
                  path_to_save= None):
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=figsize)   
    for cluster in range(model.n_clusters):
        plt.scatter(coordinates[predictions==cluster,0], 
                    coordinates[predictions==cluster,1],
                    s=50, 
                    c=colors[cluster],
                    marker="o", 
                    edgecolor="black",
                    label=f"Cluster {cluster}")
        plt.title("Clusters plot")
        plt.legend()
    plt.scatter(model.cluster_centers_[:, 0], 
                model.cluster_centers_[:, 1],
                s=250, 
                marker='*',
                c='red', 
                edgecolor='black',
                label='centroids')
    plt.show()
    if path_to_save:
        plt.savefig(join(path_to_save, "cluster_plot.jpg"))
    plt.close()
    
    
def plot_clusters_profile(data, 
                          model, 
                          predictions, 
                          dtype_predictions, 
                          figsize, 
                          title_fontsize,
                          fontsize, 
                          path_to_save):
    for cluster in range(model.n_clusters):
        print(f"Cluster {cluster} is being treated...")
        cluster_samples_idx = np.where(predictions == cluster)
        data_clustered = data.iloc[cluster_samples_idx[0],:]
        fig, axs = plt.subplots(len(data_clustered.columns), figsize=figsize)
        title = f"Cluster {cluster} - Sample size: {len(data_clustered)}/{len(data)}"
        fig.suptitle(title, fontsize=title_fontsize)
        for number, column_name in enumerate(data_clustered.columns):
            if dtype_predictions.loc[column_name].any() == "categorical":
                freq = data_clustered[column_name].value_counts()
                total = data_clustered[column_name].value_counts().sum()
                res = (freq/total)*100 
                axs[number].set_ylabel(column_name, fontsize = fontsize)
                res.plot(kind='pie', 
                         ax=axs[number], 
                         autopct='%1.1f%%', 
                         startangle=90, 
                         shadow=False, 
                         legend=False, 
                         fontsize=fontsize)
            else:
                mean  = data_clustered[column_name].mean().round(2)
                median = data_clustered[column_name].median().round(2)
                axs[number].set_title(f"mean:{mean}\nmedian:{median}", fontsize=fontsize)
                data_clustered[column_name].plot(kind="box", ax = axs[number], fontsize=fontsize)
        plt.savefig(join(path_to_save, f"Cluster {cluster}.jpg"))
        plt.close(fig)
    print("DONE!")
