from .CPAlgorithm import *
import numba
from joblib import Parallel, delayed


class BE(CPAlgorithm):
    """Borgatti Everett algorithm.

    An algorithm for finding single core-periphery pair in networks.

    Parameters
    ----------
    num_runs : int
           Number of runs of the algorithm (optional, default: 10)
           Run the algorithm num_runs times. Then, this algorithm outputs the result yielding the maximum quality.

    Examples
    --------
    Create this object.

    >>> import cpnet
    >>> be = cpnet.BE()

    **Core-periphery detection**

    Detect core-periphery structure in network G (i.e., NetworkX object):

    >>> be.detect(G)

    Retrieve the ids of the core-periphery pair to which each node belongs:

    >>> pair_id = be.get_pair_id()

    Retrieve the coreness:

    >>> coreness = be.get_coreness()

    .. note::

       This algorithm accepts unweighted and undirected networks only.
       Also, the algorithm assigns all nodes into the same core-periphery pair by construction, i.e., c[node_name] =0 for all node_name.
       This algorithm is stochastic, i.e., one would obtain different results at each run.

    .. rubric:: Reference

    [1] S. P. Borgatti and M. G. Everett. Models of core/periphery structures. Soc.~Netw., 21(4):375–395, 2000.

    """

    def __init__(self, num_runs=10):
        self.num_runs = num_runs
        self.n_jobs = 1

    def detect(self, G):
        """Detect a single core-periphery pair using the Borgatti-Everett algorithm.

        Parameters
        ----------
        G : NetworkX graph object

        Examples
        --------
        >>> import networkx as nx
        >>> import cpnet
        >>> G = nx.karate_club_graph()  # load the karate club network.
        >>> be = cpnet.BE()
        >>> be.detect(G)

        """

        A, nodelabel = utils.to_adjacency_matrix(G)

        def _detect(A_indptr, A_indices, A_data, num_nodes):
            x = _kernighan_lin_(A_indptr, A_indices, A_data, num_nodes)
            x = x.astype(int)
            cids = np.zeros(num_nodes).astype(int)
            Q, qs = _score_(A_indptr, A_indices, A_data, cids, x, num_nodes)
            return {"cids": cids, "x": x, "q": Q}

        res = Parallel(n_jobs=self.n_jobs)(
            delayed(_detect)(A.indptr, A.indices, A.data, A.shape[0])
            for i in range(self.num_runs)
        )
        res = max(res, key=lambda x: x["q"])
        cids, x, Q = res["cids"], res["x"], res["q"]

        self.nodelabel = nodelabel
        self.c_ = cids.astype(int)
        self.x_ = x.astype(int)
        self.Q_ = Q
        self.qs_ = [Q]

    def _score(self, A, c, x):
        num_nodes = A.shape[0]
        Q, qs = _score_(A.indptr, A.indices, A.data, c, x, num_nodes)
        return qs


@numba.jit(nopython=True, cache=True)
def _kernighan_lin_(A_indptr, A_indices, A_data, num_nodes):

    M = np.sum(A_data) / 2
    p = M / np.maximum(1, (num_nodes * (num_nodes - 1) / 2))
    x = np.zeros(num_nodes)
    Nperi = num_nodes
    for i in range(num_nodes):
        if np.random.rand() < 0.5:
            x[i] = 1
            Nperi -= 1

    xt = x.copy()
    xbest = x.copy()
    fixed = np.zeros(num_nodes)
    Dperi = np.zeros(num_nodes)

    for j in range(num_nodes):

        fixed *= 0
        Nperi = 0
        numer = 0
        for i in range(num_nodes):
            Nperi += 1 - x[i]
            Dperi[i] = 0
            neighbors = A_indices[A_indptr[i] : A_indptr[i + 1]]
            for k, neighbor in enumerate(neighbors):
                Dperi[i] += 1 - x[neighbor]
                numer += x[i] + x[neighbor] - x[i] * x[neighbor]
        numer = numer / 2.0 - p * (
            (num_nodes * (num_nodes - 1.0)) / 2.0 - Nperi * (Nperi - 1.0) / 2.0
        )
        pb = 1 - Nperi * (Nperi - 1) / np.maximum(1, (num_nodes * (num_nodes - 1)))
        if np.abs(pb - 1) < 1e-8 or np.abs(pb) < 1e-8:
            Qold = 0
        else:
            Qold = numer / np.maximum(1e-20, np.sqrt(pb * (1 - pb)))

        dQ = 0
        dQmax = -np.inf
        nid = 0
        for i in range(num_nodes):
            qmax = -np.inf

            # select a node of which we update the label
            numertmp = numer
            for k in range(num_nodes):
                if fixed[k] == 1:
                    continue
                dnumer = (Dperi[k] - p * (Nperi - (1 - xt[k]))) * (2 * (1 - xt[k]) - 1)
                newNperi = Nperi + 2 * xt[k] - 1
                pb = 1.0 - (newNperi * (newNperi - 1.0)) / np.maximum(
                    1, (num_nodes * (num_nodes - 1.0))
                )
                if np.abs(pb - 1) < 1e-8 or np.abs(pb) < 1e-8:
                    q = 0
                else:
                    q = (numer + dnumer) / np.maximum(1e-20, np.sqrt(pb * (1 - pb)))
                if (qmax < q) and (pb * (1 - pb) > 0):
                    nid = k
                    qmax = q
                    numertmp = numer + dnumer
            numer = numertmp
            Nperi += 2 * xt[nid] - 1
            neighbors = A_indices[A_indptr[i] : A_indptr[i + 1]]
            for k, neik in enumerate(neighbors):
                Dperi[neik] += 2 * xt[nid] - 1
            xt[nid] = 1 - xt[nid]
            dQ = dQ + qmax - Qold
            Qold = qmax

            # Save the core-periphery pair if it attains the largest quality
            if dQmax < dQ:
                xbest = xt.copy()
                dQmax = dQ
            fixed[nid] = 1
        if dQmax <= 1e-7:
            break
        xt = xbest.copy()
        x = xbest.copy()
    return xbest


@numba.jit(nopython=True, cache=True)
def _score_(A_indptr, A_indices, A_data, _c, _x, num_nodes):

    M = 0.0
    pa = 0
    pb = 0
    nc = 0
    mcc = 0
    for i in range(num_nodes):
        nc += _x[i]

        neighbors = A_indices[A_indptr[i] : A_indptr[i + 1]]
        for k, j in enumerate(neighbors):
            mcc += _x[i] + _x[j] - _x[i] * _x[j]
            M += 1

    mcc = mcc / 2
    M = M / 2
    M_b = (nc * (nc - 1) + 2 * nc * (num_nodes - nc)) / 2
    pa = M / np.maximum(1, num_nodes * (num_nodes - 1) / 2)
    pb = M_b / np.maximum(1, num_nodes * (num_nodes - 1) / 2)

    Q = (mcc - pa * M_b) / np.maximum(
        1e-20, (np.sqrt(pa * (1 - pa)) * np.sqrt(pb * (1 - pb)))
    )
    Q = Q / np.maximum(1, (num_nodes * (num_nodes - 1) / 2))

    if Q > 1:
        Q = 1
    if Q < -1:
        Q = -1

    return Q, [Q]
