import numpy as np
import networkx as nx
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import norm
from scipy import sparse


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
    num_of_thread=4,
    num_of_rand_net=500,
    q_tilde=[],
    s_tilde=[],
):
    """(q,s)-test for core-periphery structure.

    This function computes the significance of individual core-periphery pairs using either the Erdos-Renyi or the configuration model as the null model.

    Parameters
    ----------
    pair_id : dict
	keys and values of which are node names and IDs of core-periphery pairs, respectively.

    coreness : dict
	keys and values of which are node names and coreness, respectively.

    G : NetworkX graph object

    cpa : CPAlgorithm class object
	Core-periphery detection algorithm

    significance_level : float
	Significance level (optional, default 0.5)

    null_model : function
	Null model for generating randomised networks.
       	Provide either config_model or erdos_renyi (optional, default config_model).
       	One can use another null models.
       	Specifically, one needs to define a function taking NetworkX graph object as input and randomised network as its output.
       	Then, one gives the defined function, say myfunc,  to qstest by null_model=myfunc.

    sfunc : function
	Size function (optional, default sz_n)
       In the (q,s)--test, one is required to provide a function for measuring the size of an individual core-periphery pair. By default, this function is the number of nodes in the core-periphery pair (i.e., sz_n). One can set sz_degree, which measures the size as the sum of the degree of nodes belonging to the core-periphery pair.

    num_of_thread : function
	Number of thread (optional, default 4)

    	The (q,s)--test uses multiple threads to compute the significance.

    num_of_rand_net : int
	Number of randomised networks (optional, default 500)

    Returns
    -------
    sig_pair_id : dict
	keys and values of which are node names and IDs of core-periphery pairs, respectively. If nodes belong to insignificant core-periphery pair, then the values are None.

    sig_coreness : dict
	significance[i] = True or significance[i] = False indicates core-periphery pair i is significant or insignificant, respectively. If nodes belong to insignificant core-periphery pair, then the values are None.

    significance : list
	significance[i] = True or significance[i] = False indicates core-periphery pair i is significant or insignificant, respectively.

    p_values : list
	p_values[i] is the p-value of core-periphery pair i.

    Examples
    --------
    Detect core-periphery pairs in the karate club network.

    >>> import cpnet
    >>> km = cpnet.KM_config()
    >>> km.detect(G)
    >>> pair_id = km.get_pair_id()
    >>> coreness = km.get_coreness()

    Examine the significance of each core-periphery pair using the configuration model:

    >>> sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(pair_id, coreness, G, km)

    or

    >>> sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(pair_id, coreness, G, km, null_model=config_model)

    Examine the significance of each core-periphery pair using the Erdos-Renyi random graph:

    >>>  sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(pair_id, coreness, G, km, null_model=erdos_renyi)

    .. rubric:: Reference

    Sadamori Kojaku and Naoki Masuda.
    A generalised significance test for individual communities in networks.
    Scientific Reports, 8:7351 (2018)
    """
    A = nx.adjacency_matrix(G)
    nodelabels = G.nodes()
    pair_id_a = np.array([pair_id[x] for x in nodelabels])
    coreness_a = np.array([coreness[x] for x in nodelabels])

    q = np.array(cpa.score(G, pair_id, coreness), dtype=np.float)
    s = np.array(sfunc(A, pair_id_a, coreness_a), dtype=np.float)
    C = len(q)
    alpha_corrected = 1.0 - (1.0 - significance_level) ** (1.0 / float(C))

    if len(q_tilde) == 0:
        results = Parallel(n_jobs=num_of_thread)(
            delayed(sampling)(G, cpa, sfunc, null_model)
            for i in tqdm(range(num_of_rand_net))
        )
        if isinstance(results[0]["q"], list):
            q_tilde = np.array(sum([res["q"] for res in results], []))
        else:
            q_tilde = np.concatenate([res["q"] for res in results])
        if isinstance(results[0]["s"], list):
            s_tilde = np.array(sum([res["s"] for res in results], []))
        else:
            s_tilde = np.concatenate([res["s"] for res in results])

    q_ave = np.mean(q_tilde)
    s_ave = np.mean(s_tilde)
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
        w = np.exp(-(((s[cid] - s_tilde) / (np.sqrt(2.0) * h * s_std)) ** 2))
        cd = norm.cdf(
            (
                (q[cid] - q_tilde) / (h * q_std)
                - gamma * (s[cid] - s_tilde) / (h * s_std)
            )
            / np.sqrt(1.0 - gamma * gamma)
        )
        denom = sum(w)
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
