import copy

import warnings
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import norm
from tqdm import tqdm


def sz_n(network, c, x):
    return np.bincount(c).tolist()


def sz_degree(network, c, x):
    degree = np.array(np.sum(network, axis=1)).reshape(-1)
    return np.bincount(c, weights=degree).tolist()


def config_model(G):
    deg = [d[1] for d in G.degree()]
    return nx.expected_degree_graph(deg)
    # return nx.configuration_model(deg)


def erdos_renyi(G):
    n = G.number_of_nodes()
    p = nx.density(G)
    return nx.fast_gnp_random_graph(n, p)


def sampling(G, cpa, sfunc, null_model):
    Gr = null_model(G)
    Ar = sparse.csr_matrix(nx.adjacency_matrix(Gr))
    cpa.detect(Ar)
    q_rand = cpa.qs_
    s_rand = sfunc(Ar, cpa.c_, cpa.x_)
    return {"q": q_rand, "s": s_rand}


def qstest(
    pair_id,
    coreness,
    G,
    cpa,
    significance_level=0.05,
    null_model=config_model,
    sfunc=sz_n,
    num_of_rand_net=100,
    q_tilde=[],
    s_tilde=[],
    **params
):
    """(q,s)-test for core-periphery structure.

    Sadamori Kojaku and Naoki Masuda. A generalised significance test for individual communities in networks. Scientific Reports, 8:7351 (2018)

    :param pair_id: node i belongs to pair pair_id[i]
    :type pair_id: dict
    :param coreness: node i is a core (x[i]=1) or a periphery (x[i]=0)
    :type coreness: dict
    :param G: Network
    :type G: networkx.Graph or scipy sparse martix
    :param cpa: algorithm that detects the core-periphery structure in question
    :type cpa: CPAlgorithm
    :param significance_level: Significicance level, defaults to 0.05
    :type significance_level: float, optional
    :param null_model: funcion to produce a null model, defaults to config_model
    :type null_model: func, optional
    :param sfunc: Size function that calculates the size of a community, defaults to sz_n
    :type sfunc: func, optional
    :param num_of_thread: Number of threads, defaults to 4
    :type num_of_thread: int, optional
    :param num_of_rand_net: Number of random networks, defaults to 300
    :type num_of_rand_net: int, optional
    :param q_tilde: pre-computed sampled of strength q of core-periphery structure, defaults to []
    :type q_tilde: list, optional
    :param s_tilde: pre-computed sample of the size of a core-periphery pair, defaults to []
    :type s_tilde: list, optional

    .. highlight:: python
    .. code-block:: python

        >>> import cpnet
        >>> km = cpnet.KM_config()
        >>> km.detect(G)
        >>> pair_id = km.get_pair_id()
        >>> coreness = km.get_coreness()
        >>> sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(pair_id, coreness, G, km)
    """

    if "num_of_thread" in params:
        warnings.warn("'num_of_thread keyword' is duplicated due to a compatibility issue with numba. Only one CPU will be used.")

    A = nx.adjacency_matrix(G)
    nodelabels = G.nodes()
    pair_id_a = np.array([pair_id[x] for x in nodelabels])
    coreness_a = np.array([coreness[x] for x in nodelabels])

    q = np.array(cpa.score(G, pair_id, coreness), dtype=float)
    s = np.array(sfunc(A, pair_id_a, coreness_a), dtype=float)
    C = len(q)
    alpha_corrected = 1.0 - (1.0 - significance_level) ** (1.0 / float(C))

    if len(q_tilde) == 0:
        results = []
        for _ in tqdm(range(num_of_rand_net)):
            results+=[sampling(G, cpa, sfunc, null_model)]
        if isinstance(results[0]["q"], list):
            q_tilde = np.array(sum([res["q"] for res in results], []))
        else:
            q_tilde = np.concatenate([res["q"] for res in results])
        if isinstance(results[0]["s"], list):
            s_tilde = np.array(sum([res["s"] for res in results], []))
        else:
            s_tilde = np.concatenate([res["s"] for res in results])

    q_std = np.std(q_tilde, ddof=1)
    s_std = np.std(s_tilde, ddof=1)

    if (s_std <= 1e-30) or (q_std <= 1e-30):
        gamma = 0.0
        s_std = 1e-20
    else:
        gamma = np.corrcoef(q_tilde, s_tilde)[0, 1]

    h = float(len(q_tilde)) ** (-1.0 / 6.0)
    p_values = [1.0] * C
    significant = [False] * C

    cidx = 0
    cid2newcid = -np.ones(C).astype(int)
    for cid in range(C):
        if (s_std <= 1e-30) or (q_std <= 1e-30):
            continue
        logw = -(((s[cid] - s_tilde) / (np.sqrt(2.0) * h * s_std)) ** 2)
        cd = norm.cdf(
            (
                (q[cid] - q_tilde) / (h * q_std)
                - gamma * (s[cid] - s_tilde) / (h * s_std)
            )
            / np.sqrt(1.0 - gamma * gamma)
        )
        ave_logw = np.mean(logw)
        denom = sum(np.exp(logw - ave_logw))
        logw = logw - ave_logw
        w = np.exp(logw)
        if denom <= 1e-30:
            continue
        p_values[cid] = 1.0 - (sum(w * cd) / denom)
        significant[cid] = p_values[cid] <= alpha_corrected

        if significant[cid]:
            cid2newcid[cid] = cidx
            cidx += 1

    sig_pair_id = copy.deepcopy(pair_id)
    sig_coreness = copy.deepcopy(coreness)

    for k, v in sig_pair_id.items():
        if significant[v]:
            sig_pair_id[k] = cid2newcid[pair_id[k]]
        else:
            sig_pair_id[k] = None
            sig_coreness[k] = None

    return sig_pair_id, sig_coreness, significant, p_values
