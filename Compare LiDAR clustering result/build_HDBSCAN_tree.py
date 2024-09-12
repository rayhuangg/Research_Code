
#%%
import numpy as np
import seaborn as sns
import hdbscan
import matplotlib.pyplot as plt
import sklearn.datasets as data

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}

moons, _ = data.make_moons(n_samples=80, noise=0.05)
blobs, _ = data.make_blobs(n_samples=100, centers=[(-0.75,2), (2.0, 2.0)], cluster_std=0.25)
X = np.vstack([moons, blobs])
plt.figure(figsize=(12,8), dpi=300)
plt.scatter(X[:,0], X[:,1])
plt.xticks([])
plt.yticks([])
plt.savefig('raw_data.png',dpi=300, bbox_inches='tight', transparent=True)

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(X)

# Retrieve the cluster labels from the fitted model
hdbscan_labels = clusterer.labels_


# HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    # gen_min_span_tree=True, leaf_size=40, memory=Memory(None),
    # metric='euclidean', min_cluster_size=5, min_samples=None, p=None)
plt.figure(figsize=(12,8), dpi=300)

clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=80,
                                      edge_linewidth=2)
plt.savefig('minimum_spanning_tree.png',dpi=300, bbox_inches='tight', transparent=True)
#%%
plt.figure(figsize=(12,8), dpi=300)
clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
plt.savefig('single_linkage_tree.png',dpi=300, bbox_inches='tight', transparent=True)
#%%
plt.figure(figsize=(12,8), dpi=300)
clusterer.condensed_tree_.plot()

#%%
plt.figure(figsize=(12,8), dpi=300)
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
plt.savefig('condensed_tree.png',dpi=300, bbox_inches='tight', transparent=True)

# %%
plt.figure(figsize=(12,8), dpi=300)
plt.scatter(X[:, 0], X[:, 1],c=hdbscan_labels, cmap='viridis', s=50)
plt.xticks([])
plt.yticks([])
plt.savefig('hdbscan_cluster.png',dpi=300, bbox_inches='tight', transparent=True)



# %%
