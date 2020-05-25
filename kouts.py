#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.datasets.samples_generator import make_circles
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
import networkx as nx
import seaborn as sns
sns.set()


# In[22]:


G=nx.read_edgelist('C:\Users\User\Desktop\New folder\email-Eu-core.txt',create_using=nx.Graph(),nodetype=int)


for x in range(G.number_of_nodes()):
    e=(x,x)
    if(G.has_edge(*e)):
        G.remove_edge(*e)
#G=nx.Graph()
#G.add_edge(1,2)
#G.add_edge(1,3)
#G.add_edge(1,5)
#G.add_edge(2,3)
#G.add_edge(3,4)
#G.add_edge(5,4)
#G.add_edge(6,5)
#G.add_edge(6,4)
print nx.info(G)


# In[16]:


def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=5, alpha=1)


# In[17]:


draw_graph(G)
W = nx.adjacency_matrix(G)
print(W.todense())


# In[18]:


# degree matrix
D = np.diag(np.sum(np.array(W.todense()), axis=1))
print('degree matrix:')
print(D)
# laplacian matrix
L = D - W
print('laplacian matrix:')
print(L)


# In[10]:


e, v = np.linalg.eig(L)
# eigenvalues
print('eigenvalues:')
print(e)
# eigenvectors
print('eigenvectors:')
print(v)



# In[11]:


max=0
k=0
for x in range(1,len(e)):
    val=e[x]-e[x-1]
    val=abs(val)
    if(max<val):
        max=val
        k=x
print(k)


# In[12]:


from sklearn.cluster import KMeans
n_clusters=k
kmeans = KMeans(n_clusters)
kmeans.fit(v[:,1:n_clusters])
colors = kmeans.labels_


print("Clusters:", colors)


# In[ ]:





# In[ ]:




