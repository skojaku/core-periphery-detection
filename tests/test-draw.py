# %%
%load_ext autoreload
%autoreload 2

import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from scipy import sparse, stats

import cpnet

# %%
net = sparse.load_npz("net.npz")
node_table = pd.read_csv("node.csv", sep="\t").rename(columns={"Unnamed: 0": "id"})
#G = nx.karate_club_graph()
G = nx.from_scipy_sparse_matrix(net)

# Core-periphery detection
model = cpnet.KM_ER()
model.detect(G)
coreness = model.get_coreness()
pair_ids = model.get_pair_id()

# Statistical test
#sig_c, sig_x, significant, p_values = cpnet.qstest(
#    pair_ids, coreness, G, model, significance_level=0.01,num_of_thread = 30
#)

# %%
# Plot
fig, ax = plt.subplots(figsize=(5,5))
a = cpnet.draw(G, pair_ids, coreness, ax, max_group_num = 2)
#a = cpnet.draw(G, coreness, pair_ids, ax, max_group_num = 2)

# %%
