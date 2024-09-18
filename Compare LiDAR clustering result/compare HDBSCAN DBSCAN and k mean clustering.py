#%%

import numpy as np
import random

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans

import hdbscan

random_seed = 31026
np.random.seed(random_seed)
random.seed(random_seed)

moons, _ = make_moons(n_samples=80, noise=0.05)
blobs, _ = make_blobs(n_samples=100, centers=[(-0.75,2), (2.0, 2.0)], cluster_std=0.25)
X = np.vstack([moons, blobs])
X_DBSCAN = X.copy()
X_HDBSCAN = X.copy()
X_KMEANS = X.copy()

# DBSCAN聚類
dbscan = DBSCAN(eps=0.65, min_samples=20)
dbscan_labels = dbscan.fit_predict(X_DBSCAN)

# HDBSCAN聚類
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
hdbscan_labels = hdbscan_clusterer.fit_predict(X_HDBSCAN)

# K-Means聚類
kmeans_clusters = 4
kmeans = KMeans(n_clusters=kmeans_clusters)
kmeans_labels = kmeans.fit_predict(X_KMEANS)
# 繪製結果
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# K-Means結果
axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].text(0.5, 0.95, f'n_clusters={kmeans_clusters}', transform=axes[0].transAxes, fontsize=15,
             verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

# DBSCAN結果
axes[1].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', s=50)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].text(0.5, 0.95, f'eps=0.65\nMinPts=20', transform=axes[1].transAxes, fontsize=15,
             verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

# HDBSCAN結果
axes[2].scatter(X[:, 0], X[:, 1], c=hdbscan_labels, cmap='viridis', s=50)
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[2].text(0.5, 0.95, f'MinClusterSize=5', transform=axes[2].transAxes, fontsize=15,
             verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))


# 儲存圖片
plt.savefig('clustering_result_DBSCAN_HDBSCAN_Kmeans.png', dpi=300, bbox_inches='tight', transparent=True)

# %%
fig = plt.figure(figsize=(8, 7))
plt.scatter(X[:, 0], X[:, 1], cmap='viridis', s=50)
plt.xticks([])
plt.yticks([])
plt.savefig('clustering_raw.png', dpi=300, bbox_inches='tight', transparent=True)

# %%
fig = plt.figure(figsize=(8, 7))
plt.scatter(X[:, 0], X[:, 1],c=hdbscan_labels, cmap='viridis', s=50)
plt.xticks([])
plt.yticks([])
plt.savefig('clustering_hdbscan.png', dpi=300, bbox_inches='tight', transparent=True)
