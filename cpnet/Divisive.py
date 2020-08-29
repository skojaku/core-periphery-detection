from .CPAlgorithm import *
from .BE import BE
from . import utils
from scipy import sparse
import numba


class Divisive(CPAlgorithm):
    """Divisive algorithm.

    An algorithm for finding multiple core-periphery pairs in networks.
    This algorithm partitions a network into communities using the Louvain algorithm.
    Then, it partitions each community into a core and a periphery using the BE algorithm.
    The quality of a community is computed by that equipped with the BE algorithm.

    Parameters
    ----------
    num_runs : int
        Number of runs of the algorithm (optional, default: 10)
        Run the algorithm num_runs times. Then, this algorithm outputs the result yielding the maximum quality.

    Examples
    --------
    Create this object.

    >>> import cpnet
    >>> dv = cpnet.Divisive()

    **Core-periphery detection**

    Detect core-periphery structure in network G (i.e., NetworkX object):

    >>> dv.detect(G)

    Retrieve the ids of the core-periphery pair to which each node belongs:

    >>> pair_id = dv.get_pair_id()

    Retrieve the coreness:

    >>> coreness = dv.get_coreness()

    .. note::

       This algorithm accepts unweighted and undirected networks only.
       This algorithm is stochastic, i.e., one would obtain different results at each run.

    .. rubric:: Reference

        [1] S. Kojaku and N. Masuda. Core-periphery structure requires something else in the network. New Journal of Physics, 20(4):43012, 2018

    """

    def __init__(self, num_runs=10):
        self.num_runs = num_runs

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
        >>> dv = cpnet.Divisive()
        >>> dv.detect(G)
        """

        A, nodelabel = utils.to_adjacency_matrix(G)

        be = BE()

        # divide a network into communities
        cids = _louvain_(A)

        x = []
        q = []
        K = int(np.max(cids) + 1)
        for c in range(K):
            nodes = np.where(cids == c)[0]
            As = A[nodes, :][:, nodes]
            be.detect(As)
            x += [be.x_[k] for k, node in enumerate(nodes)]
            q += [be.score(As, be.c_, be.x_)[0]]

        x = np.array(x)
        self.nodelabel = nodelabel
        self.c_ = cids.astype(int)
        self.x_ = x.astype(int)
        self.Q_ = np.sum(q)
        self.qs_ = q

    def _score(self, A, c, x):
        N = A.shape[0]
        be = BE()
        q = []
        for cid in range(np.max(c) + 1):
            nodes = np.where(c == cid)[0]
            As = A[nodes, :][:, nodes]
            q += [be.score(As, np.ones(nodes.size), x[nodes])[0]]
        return q


@numba.jit(nopython=True, cache=True)
def _label_switching_(A_indptr, A_indices, A_data, num_nodes, alpha=0.5):

    deg = np.zeros(num_nodes)
    D = np.zeros(num_nodes)
    selfloop = np.zeros(num_nodes)
    cids = np.arange(num_nodes)
    for nid in range(num_nodes):
        neighbors = A_indices[A_indptr[nid] : A_indptr[nid + 1]]
        weight = A_data[A_indptr[nid] : A_indptr[nid + 1]]
        deg[nid] = np.sum(weight)
        D[cids[nid]] += deg[nid]
        selfloop[nid] = np.sum(weight[neighbors == nid])

    M = np.sum(deg) / 2
    while True:
        order = np.random.choice(num_nodes, size=num_nodes, replace=False)
        updated_node_num = 0

        for k, node_id in enumerate(order):

            # Get the weight and normalized weight
            neighbors = A_indices[A_indptr[node_id] : A_indptr[node_id + 1]]
            weight = A_data[A_indptr[node_id] : A_indptr[node_id + 1]]

            # Calculate the grain
            clist = np.unique(cids[neighbors])
            next_cid = -1
            qself = 0
            dqmax = 0
            for cprime in clist:
                neis = neighbors[cids[neighbors] == cprime]
                dq = (1 - alpha) * np.sum(weight[cids[neighbors] == cprime])

                D_prime = D[cprime] - deg[node_id] * (cprime == cids[node_id])
                dq -= alpha * D_prime / (2.0 * M)
                dq += 0.5 * (selfloop[node_id] - deg[node_id] * deg[node_id] / (2 * M))

                if cprime == cids[node_id]:
                    qself = dq
                    continue

                if dqmax < dq:
                    next_cid = cprime
                    dqmax = dq

            dqmax -= qself
            if dqmax <= 1e-16:
                continue

            D[cids[node_id]] -= deg[node_id]
            D[next_cid] += deg[node_id]

            cids[node_id] = next_cid
            updated_node_num += 1

        if (updated_node_num / num_nodes) < 1e-3:
            break

    return cids


@numba.jit(nopython=True, cache=True)
def _score_(A_indptr, A_indices, A_data, _c, num_nodes, alpha=0.5):
    cids = np.unique(_c)
    K = int(np.max(cids) + 1)
    q = np.zeros(K)
    D = np.zeros(num_nodes)
    deg = np.zeros(num_nodes)
    selfloop = np.zeros(num_nodes)
    rho = 0
    doubleM = 0
    for i in range(num_nodes):
        neighbors = A_indices[A_indptr[i] : A_indptr[i + 1]]
        weight = A_data[A_indptr[i] : A_indptr[i + 1]]
        deg[i] = np.sum(weight)
        selfloop[i] = np.sum(weight[neighbors == i])

        q[_c[neighbors]] += weight
        D[_c[i]] += deg[i]
        doubleM += np.sum(weight)

    rho = doubleM / (num_nodes * num_nodes)
    Q = 0
    for k in range(K):
        q[k] = (1 - alpha) * q[k] - alpha * (D[k] * D[k]) / doubleM
        q[k] /= doubleM
        Q += q[k]

    return Q, q


def _louvain_(A):
    N = A.shape[0]
    C = sparse.diags(np.ones(N))
    newA = A
    Ct = C.copy()
    prev_size = N
    Qbest, _ = _score_(A.indptr, A.indices, A.data, np.ones(N).astype(int), N)

    while True:

        prev_size = newA.shape[0]

        zids = _label_switching_(newA.indptr, newA.indices, newA.data, newA.shape[0])

        _, zids = np.unique(zids, return_inverse=True)
        Zt = sparse.csr_matrix(
            (np.ones_like(zids), (np.arange(zids.size), zids)),
            shape=(zids.size, int(np.max(zids) + 1)),
        )

        Qt, _ = _score_(newA.indptr, newA.indices, newA.data, zids, newA.shape[0])

        newA = Zt.T @ newA @ Zt

        if newA.shape[0] == prev_size:
            break

        Ct = Ct @ Zt

        if Qt > Qbest:
            C = Ct
            Qbest = Qt
    cids = np.array((Ct @ sparse.diags(np.arange(Ct.shape[1]))).sum(axis=1)).reshape(-1)
    _, cids = np.unique(cids, return_inverse=True)

    return cids
