import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl

gall_feat = np.load('/root/UNICORN/vis/gall_feat.npy')
gall_label = np.load('/root/UNICORN/vis/gall_label.npy')

query_feat = np.load('/root/UNICORN/vis/query_feat.npy')
query_label = np.load('/root/UNICORN/vis/query_label.npy')

label = np.concatenate([gall_label, query_label], axis=0)

unique_label = np.unique(label)

new_gall_label = np.searchsorted(unique_label, gall_label)
new_query_label = np.searchsorted(unique_label, query_label)

gall_feat = gall_feat[new_gall_label < 10]
query_feat = query_feat[new_query_label < 10]

new_gall_label = new_gall_label[new_gall_label < 10]
new_query_label = new_query_label[new_query_label < 10]

new_gall_label = new_gall_label / 10
new_query_label = new_query_label / 10

feat = np.concatenate([gall_feat, query_feat], axis=0)

tsne = TSNE(n_components=2, init='pca')

transformed_feat = tsne.fit_transform(feat)

transformed_gall_feat = transformed_feat[:new_gall_label.shape[0]]
transformed_query_feat = transformed_feat[new_gall_label.shape[0]:]

transformed_gall_feat = transformed_gall_feat - transformed_feat.min(axis=0)[np.newaxis,:]
transformed_query_feat = transformed_query_feat - transformed_feat.min(axis=0)[np.newaxis,:]
transformed_feat = transformed_feat - transformed_feat.min(axis=0)[np.newaxis,:]

transformed_gall_feat = transformed_gall_feat / transformed_feat.max(axis=0)[np.newaxis,:]
transformed_query_feat = transformed_query_feat / transformed_feat.max(axis=0)[np.newaxis,:]

fig, ax = plt.subplots(dpi=250)

ax.scatter(transformed_gall_feat[:,0], transformed_gall_feat[:,1], c=new_gall_label, cmap='tab10', marker = 'o')
ax.scatter(transformed_query_feat[:,0], transformed_query_feat[:,1], c=new_query_label, cmap='tab10', marker = 's')
    
plt.savefig("/root/UNICORN/vis/tsne_img.jpg")
