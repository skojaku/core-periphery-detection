# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats

import cpnet

# %%
net = sparse.load_npz("net.npz")
node_table = pd.read_csv("node.csv", sep="\t").rename(columns={"Unnamed: 0": "id"})


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
