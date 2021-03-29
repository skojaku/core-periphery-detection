import os
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

# Load the blog net
edges = pd.read_csv(
    "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/out.moreno_blogs_blogs"
)
node_table = pd.read_csv(
    "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/ent.moreno_blogs_blogs.blog.orientation",
    sep=",",
    header=None,
    names=["class"],
)
N = np.max(edges.max().values) + 1
net = sparse.csc_matrix(
    (np.ones(edges.shape[0]), (edges.source - 1, edges.target - 1)), shape=(N, N)
)

# Extract the largest connected component
G = nx.from_scipy_sparse_matrix(net)
node_set = sorted(nx.connected_components(G), key=len, reverse=True)[0]
G = G.subgraph(node_set)
node_table = node_table.iloc[np.array(list(node_set)), :]

# Convert to the adjacency matrix
net = nx.adjacency_matrix(G)
net = net + net.T
net.data = np.ones_like(net.data)

qth = 0.1
deg = np.array(net.sum(axis=0)).reshape(-1)
is_hub_nodes = ["Core" if l else "Periphery" for l in deg >= np.quantile(deg, 1 - qth)]
node_table = pd.DataFrame(
    {"deg": deg, "is_hub_nodes": is_hub_nodes, "community": node_table["class"]}
)

node_table.to_csv("node.csv", sep="\t")
sparse.save_npz("net.npz", net)
