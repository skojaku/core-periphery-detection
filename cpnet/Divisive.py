import numba
import numpy as np
from scipy import sparse

from . import utils
from .BE import BE
from .CPAlgorithm import CPAlgorithm


class Divisive(CPAlgorithm):
    """Divisive algorithm.

    This algorithm partitions a network into communities using the Louvain algorithm.
    Then, it partitions each community into a core and a periphery using the BE algorithm.
    The quality of a community is computed by that equipped with the BE algorithm.

    S. Kojaku and N. Masuda. Core-periphery structure requires something else in the network. New Journal of Physics, 20(4):43012, 2018

    .. highlight:: python
    .. code-block:: python

        >>> import cpnet
        >>> alg = cpnet.Divisive()
        >>> alg.detect(G)
        >>> pair_id = alg.get_pair_id()
        >>> coreness = alg.get_coreness()

    .. note::

        - [x] weighted
        - [ ] directed
        - [x] multiple groups of core-periphery pairs
        - [ ] continuous core-periphery structure
    """

    def __init__(self, num_runs=10):
        """Initialize algorithm.

        :param num_runs: number of runs, defaults to 10
        :type num_runs: int, optional
        """
        self.num_runs = num_runs

    def detect(self, G):
        """Detect core-periphery structure.

        :param G: Graph
        :type G: networkx.Graph or scipy sparse matrix
        :return: None
        :rtype: None
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
        """Calculate the strength of core-periphery pairs.

        :param A: Adjacency amtrix
        :type A: scipy sparse matrix
        :param c: group to which a node belongs
        :type c: dict
        :param x: core (x=1) or periphery (x=0)
        :type x: dict
        :return: strength of core-periphery
        :rtype: float
        """
        be = BE()
        q = []
        for cid in range(np.max(c) + 1):
            nodes = np.where(c == cid)[0]
            As = A[nodes, :][:, nodes]
            q += [be.score(As, np.ones(nodes.size), x[nodes])[0]]
        return q


@numba.jit(nopython=True, cache=True)
def _label_switching_(A_indptr, A_indices, A_data, num_nodes, alpha=0.5):
    """Modularity maximization based on the label switching algorithm.

    :param A_indptr: A.indptr, where A is scipy.csr_sparse matrix
    :type A_indptr: numpy.ndarray
    :param A_indices: A.indices, where A is scipy.csr_sparse matrix
    :type A_indices: numpy.ndarray
    :param A_data: A.data, where A is scipy.csr_sparse matrix
    :type A_data: numpy.ndarray
    :param num_nodes: number of nodes
    :type num_nodes: int
    :param alpha: resolution, defaults to 0.5
    :type alpha: float, optional
    :return: group membership
    :rtype: numpy.ndarray
    """

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

        for _k, node_id in enumerate(order):

            # Get the weight and normalized weight
            neighbors = A_indices[A_indptr[node_id] : A_indptr[node_id + 1]]
            weight = A_data[A_indptr[node_id] : A_indptr[node_id + 1]]

            # Calculate the grain
            clist = np.unique(cids[neighbors])
            next_cid = -1
            qself = 0
            dqmax = 0
            for cprime in clist:
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
    doubleM = 0
    for i in range(num_nodes):
        neighbors = A_indices[A_indptr[i] : A_indptr[i + 1]]
        weight = A_data[A_indptr[i] : A_indptr[i + 1]]
        deg[i] = np.sum(weight)
        selfloop[i] = np.sum(weight[neighbors == i])

        q[_c[neighbors]] += weight
        D[_c[i]] += deg[i]
        doubleM += np.sum(weight)

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
