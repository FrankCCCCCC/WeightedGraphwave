#!/usr/bin/env python
# coding: utf-8

# <center> <h1>Using GraphWave </h1> </center>
# 
# &nbsp;
# 
# &nbsp;
# 
# The goal of the  following notebook is to show how the GraphWave algorithm can be used. 
# 
# GraphWave was implemented in Python 2.7 and requires to load the following Python packages:
# 
# + __networkx__ (for handling network objects: in particular, visualization, etc.)
# + traditional libraries for data analytics: 
#     + __seaborn__ for plotting
#     + __pandas__ for dataframes
#     + __sklearn__ for analytics
# 

# In[1]:


#get_ipython().magic(u'matplotlib inline')
import networkx as nx 
import numpy as np
import pandas as pd
import seaborn as sb
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics

import matplotlib.pyplot as plt
import graphwave
from graphwave.shapes import temp
from graphwave.graphwave import *
from graphwave import tmp
import math



np.random.seed(10)


# ## I. Creating a graph

# In[2]:


# 1- Start by defining our favorite regular structure

width_basis =399
nbTrials = 20


################################### EXAMPLE TO BUILD A SIMPLE REGULAR STRUCTURE ##########
## REGULAR STRUCTURE: the most simple structure:  basis + n small patterns of a single type

### 1. Choose the basis (cycle, torus or chain)
basis_type = "cycle" 

### 2. Add the shapes 
n_shapes =1 ## numbers of shapes to add 
#shape=["fan",6] ## shapes and their associated required parameters  (nb of edges for the star, etc)
#shape=["star",6]
list_shapes = [["house"]] * n_shapes

### 3. Give a name to the graph
identifier = 'AA'  ## just a name to distinguish between different trials
name_graph = 'houses'+ identifier
sb.set_style('white')


#figsize = 11,9
#figure, ax = plt.subplots(figsize=figsize)
#plt.subplots(figsize=figsize)
### 4. Pass all these parameters to the Graph Structure
add_edges = 0

#for i in range (294):
#    role_id.append(i)
G = nx.DiGraph()
G,b,weightlist = tmp.createGraph('football data.CSV')
G.to_directed()
role_id=b

temp.plot_networkx(G, role_id)


# (Note: best visualization of the graphs are obtained using Gephi, or some other specialized graph visualization software)

# ## II. Running GraphWave
# 
# 
# We propose here a simple demonstration of GraphWave using both the automatic version (part a) and the manual version. This shows how to use GraphWave in a parameter-free version, or giving the analyst the possibility to select an adequate scale value.
# 
# For each of these approaches, we compute the signature by calling GraphWave. We then compute its PCA projection to visualize the embeddings. Note that in this very simple examples, GraphWave recovers structura equivalence, as shown by the overlapping embeddings on the first principal components.
# 
# #### a. Multiscale GraphWave: Automatic selection of the range of scales

# In[3]:


chi, heat_print, taus = graphwave_alg(G, np.linspace(0,100,25), taus='auto', verbose=True)


# We now visualize the resulting embeddings by computing their PCA projections. We also run KMeans to assess how well the signatures that we have here generated enable the recovery of structural roles.

# In[4]:


#nb_clust = len(np.unique(role_id))
nb_clust = 10
pca = PCA(n_components=2)
trans_data = pca.fit_transform(StandardScaler().fit_transform(chi))
km = KMeans(n_clusters=nb_clust)
km.fit(trans_data)
labels_pred=km.labels_


#print metrics.completeness_score(role_id, labels_pred)


######## Params for plotting
cmapx=plt.get_cmap('rainbow')
plt.rcParams['savefig.dpi'] = 150 #图片像素
plt.rcParams['figure.dpi'] = 150 #分辨率

x=np.linspace(0,1,nb_clust+1)
col=[cmapx(xx) for xx in x ]
markers = {0:'.',1: '.'}
plt.legend(fontsize=3)

for c in np.unique(role_id):
    indc = [i for i,x in enumerate(role_id) if x==c]

    plt.scatter(trans_data[indc,0], trans_data[indc,1],
                c=np.array(col)[list(np.array(labels_pred)[indc])],
                marker=markers[c%len(markers)], s=150)



#f = open("footballss.CSV")
#lines = f.read()
#strlist0 = lines.replace(',',' ')
#strlist = lines.split()

f = open('footballss.CSV', 'r')
csvreader = csv.reader(f)
final_list = list(csvreader)

    
    
labels = role_id
for label,c, x, y in zip(labels,labels_pred, trans_data[:, 0], trans_data[:, 1]):
            plt.annotate(label,xy=(x, y), xytext=(-7, -8), textcoords='offset pixels',fontsize=5)
#for label,c, x, y in zip(final_list,labels_pred, trans_data[:, 0], trans_data[:, 1]):
#            plt.annotate(label,xy=(x, y), xytext=(-7, -5), textcoords='offset pixels',fontsize=2)
                        


#print metrics.homogeneity_score(a,b)

# #### Uniscale GraphWave: Hand-selected value for tau

# In[9]:


np.linspace(0,10,100)


# In[10]:


### Select a scale of interest (here we select a particular range of scale. See associated paper for 
### guidelines on how to select the appropriate scale.)

### Compute the heat wavelet
from graphwave.graphwave import *

time_pts = list(np.arange(0,50,0.5))
chi,heat_print, taus=graphwave_alg(G, np.linspace(0,10,100), taus=[1.0], verbose=True) 
print(chi.shape, len(time_pts))


# Note that in the EPFL implementation, by construction, the wavelet scales are all divided by the maximum eigenvalue $\lambda_N$.

# In[13]:



# ## III. Visualizing the Characteristic functions
# 
# We now propose to show how to visualize characteristic functions.
# 

# In[14]:

'''
mapping = {u: i for i,u in enumerate(np.unique(role_id))}
cmap=plt.get_cmap('gnuplot')
role_id_plot=[cmap(x) for x in np.linspace(0,1,len(np.unique(role_id)))]
plt.figure()
ind_x=range(chi[0].shape[0])[0::2]
ind_y=range(chi[0].shape[0])[1::2]
for i in np.random.choice(range(G.number_of_nodes()),10,replace=False):
    _ = plt.plot(chi[i, ind_x],chi[i, ind_y],label=str(i),color=role_id_plot[mapping[role_id[i]]])

_ = plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
'''

# In[ ]:


