#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from collections import Counter
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.datasets.samples_generator import make_circles
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
import networkx as nx
import seaborn as sns
sns.set()


# In[30]:


G=nx.read_edgelist("email-Eu-core.txt",create_using=nx.Graph(),nodetype=int)

G.remove_edges_from(G.selfloop_edges())
list(nx.isolates(G))
G.remove_nodes_from(list(nx.isolates(G)))
print (nx.info(G))


# In[31]:


def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=5, alpha=1)


# In[55]:


# fixing the size of the figure 
plt.figure(figsize =(10, 10))
nx.draw(G, with_labels=True)
W = nx.adjacency_matrix(G,nodelist=sorted(G.nodes()))
print(W.todense())
W.shape


# In[56]:


# degree matrix
D = np.diag(np.sum(np.array(W.todense()), axis=1))
print('degree matrix:')
print(D)
# laplacian matrix
L = D - W
print('laplacian matrix:')
print(L)


# In[57]:


e, v = np.linalg.eig(L)
# eigenvalues
print('eigenvalues:')
print(e)
# eigenvectors
print('eigenvectors:')
print(v)
print(len(e))
print(v.shape)


# In[67]:


from scipy.sparse import csgraph
# from scipy.sparse.linalg import eigsh
from numpy import linalg as LA
def eigenDecomposition(A, plot = True, topK = 10):
    """
    :param A: Adjacency matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors
    
    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic
    
    """
    n_components = A.shape[0]
    
    eigenvalues, eigenvectors = np.linalg.eig(L)
    
    
    if plot:
        plt.title('Largest eigenvalues of input matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.grid()
        
    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigenvalues
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1
        
    return nb_clusters, eigenvalues, eigenvectors


# In[68]:


k, _,  _ = eigenDecomposition(W)
#print(k)
print(f'Optimal number of clusters {k}')


# In[69]:


v=v.real
e=e.real
print(v)
print(e)


# In[70]:


# sort these based on the eigenvalues
print('eigenvectors:')
v = v[:,np.argsort(e)]
print(v)
print('-------')
e = e[np.argsort(e)]
print('eigenvalues:')
print(e)


# In[73]:


from sklearn.cluster import KMeans

kmeans = KMeans(103)
kmeans.fit(v[:,1:4])
colors = kmeans.labels_


# In[74]:


print(Counter(colors).values())
#print("Clusters:", colors[24])
print("Clusters:", colors)
print(len(Counter(colors).values()))
i=0
for k,v in Counter(colors).items():
    if v == 1:
        i+=1
print(i)


# In[ ]:




