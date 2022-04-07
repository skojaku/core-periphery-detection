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
G = nx.from_scipy_sparse_matrix(net)
# %%

def calc_nmi(y, ypred):
    _, r = np.unique(y, return_inverse=True)
    _, c = np.unique(ypred, return_inverse=True)
    N = np.max(r) + 1
    K = np.max(c) + 1
    W = sparse.csr_matrix((np.ones_like(c), (r, c)), shape=(N, K))
    W = W / W.sum()
    W = np.array(W.toarray())

    wcol = np.array(W.sum(axis=0)).reshape(-1)
    wrow = np.array(W.sum(axis=1)).reshape(-1)

    Irc = stats.entropy(W.reshape(-1), np.outer(wrow, wcol).reshape(-1))
    Q = 2 * Irc / (stats.entropy(wrow) + stats.entropy(wcol))
    return Q


# %%
models = {
    "KM_config": cpnet.KM_config(),
    "Modularity": cpnet.Divisive(),
    "KM_ER": cpnet.KM_ER(),
}
results = []
for name, model in models.items():

    for _i in range(30):

        model.detect(net)

        coreness = model.get_coreness()
        pair_ids = model.get_pair_id()

        x_table = (
            pd.DataFrame.from_dict(coreness, orient="index")
            .reset_index()
            .rename(columns={"index": "id", 0: "x"})
        )
        c_table = (
            pd.DataFrame.from_dict(pair_ids, orient="index")
            .reset_index()
            .rename(columns={"index": "id", 0: "c"})
        )
        cx_table = pd.merge(x_table, c_table, on="id")

        df = node_table.copy()
        df = pd.merge(df, cx_table, on="id")

        score = calc_nmi(df["community"], df["c"])
        results += [{"name": name, "score": score}]

result_table = pd.DataFrame(results)

# %%
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
ax = sns.barplot(data=result_table, x="name", y="score", palette="Blues_r")
ax.set_ylabel("Normalized Mutual Information")
ax.set_xlabel("Algorithm")
sns.despine()
# %%

#
# Visualize the detected core-periphery pairs
#

# Core-periphery detection
model = cpnet.KM_config()
model.detect(G)

coreness = model.get_coreness()
pair_ids = model.get_pair_id()

# Statistical test
sig_c, sig_x, significant, p_values = cpnet.qstest(
    pair_ids, coreness, G, model, significance_level=0.01,num_of_thread = 30
)


# %%
# Plotter
def plot_cp_pairs_matrix(c, x, labels, label2color, G, ax=None):
    # Make a node table
    x_table = (
        pd.DataFrame.from_dict(x, orient="index")
        .reset_index()
        .rename(columns={"index": "id", 0: "x"})
    )
    c_table = (
        pd.DataFrame.from_dict(c, orient="index")
        .reset_index()
        .rename(columns={"index": "id", 0: "c"})
    )
    cx_table = pd.merge(x_table, c_table, on="id")
    df = pd.DataFrame({"id":np.arange(len(labels)), "community":labels})
    df = pd.merge(df, cx_table, on="id")

    A = sparse.csr_matrix(nx.adjacency_matrix(G))
    deg = np.array(A.sum(axis = 1)).reshape(-1)
    df["deg"] = deg[df.id.values]

    # Reorder adjacency matrix
    dg = df.copy().groupby("c").apply(lambda dh: dh.sort_values(by=["x", "community"], ascending = False))
    order = dg.id.values.astype(int)
    A = A[order,:][:, order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))

    # Plot edges
    r, c, v = sparse.find(A)
    ax.scatter(x = r, y = c, s = 0.05, color = "#4d4d4d", zorder = 1, alpha = 0.5)

    # Plot community labels
    dh = 50
    for nid, (_, row) in enumerate(dg.iterrows()):
        x = -dh
        y = nid
        rect = patches.Rectangle((x, y), height = 0.01, width = dh, color = label2color[row["community"]], zorder = 2, clip_on=False)

        ax.add_patch(rect)
        x = nid
        y = -dh
        rect = patches.Rectangle((x, y), width = 1, height = dh, color = label2color[row["community"]], zorder = 2, clip_on=False)
        ax.add_patch(rect)

    N = A.shape[0]
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)

    ax.yaxis.tick_right()
    ax.invert_yaxis()
    return ax

# Prep.
cmap = sns.color_palette().as_hex()
label2color = {"left-leaning":cmap[3], "right-leaning":cmap[0]}
labels = df["community"].values

# Plot
plot_cp_pairs_matrix(sig_c, sig_x, labels, label2color, G, ax=None)
