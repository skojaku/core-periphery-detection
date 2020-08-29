import scipy as sp
import numpy as np
from .CPAlgorithm import *
from . import utils
import numba


class Surprise(CPAlgorithm):
    """ Core-periphery detection by Surprise

    Parameters
    ----------
    num_runs : int
           Number of runs of the algorithm (optional, default: 1)
           Run the algorithm num_runs times. Then, this algorithm outputs the result yielding the maximum quality.

    Examples
    --------
    Create this object.

    >>> import cpnet
    >>> spr = cpnet.Surprise()

    **Core-periphery detection**

    Detect core-periphery structure in network G (i.e., NetworkX object):

    >>> spr.detect(G)

    Retrieve the ids of the core-periphery pair to which each node belongs:

    >>> pair_id = spr.get_pair_id()

    Retrieve the coreness:

    >>> coreness = spr.get_coreness()

    .. note::

       The implemented algorithm accepts unweighted and undirected networks only.
       The algorithm finds a single CP pair in the given network, i.e., c[node_name] =0 for all node_name.
       This algorithm is stochastic, i.e., one would obtain different results at each run.

    .. rubric:: Reference

    [1] J. van Lidth de Jeude, G. Caldarelli, T. Squartini. Detecting Core-Periphery Structures by Surprise. EPL, 125, 2019

    """

    def __init__(self):
        self.num_runs = 1

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
        >>> spr = cpnet.Surprise()
        >>> spr.detect(G)
        """

        A, nodelabel = utils.to_adjacency_matrix(G)

        N = A.shape[0]
        xbest = np.zeros(N)
        qbest = 0
        for it in range(self.num_runs):
            x, q = _detect_(A.indptr, A.indices, A.data, A.shape[0])
            if q < qbest:
                xbest = x
                qbest = q

        self.nodelabel = nodelabel
        self.c_ = np.zeros(N).astype(int)
        self.x_ = x.astype(int)
        self.Q_ = qbest
        self.qs_ = [qbest]

    def _score(self, A, c, x):
        num_nodes = A.shape[0]
        qs = _score_(A.indptr, A.indices, A.data, num_nodes, x)
        return [qs]


@numba.jit(nopython=True, cache=True)
def _mat_prod_(A_indptr, A_indices, A_data, N, x):
    y = np.zeros(N)
    for i in range(N):
        neighbors = A_indices[A_indptr[i] : A_indptr[i + 1]]
        weights = A_data[A_indptr[i] : A_indptr[i + 1]]
        y[i] = np.sum(np.multiply(weights, x[neighbors]))
    return y


@numba.jit(nopython=True, cache=True)
def _detect_(A_indptr, A_indices, A_data, N):

    # ------------
    # Main routine
    # ------------
    deg = np.zeros(N)
    x = np.zeros(N)
    for i in range(N):
        if np.random.rand() < 0.5:
            x[i] = 1

        deg[i] = np.sum(A_data[A_indptr[i] : A_indptr[i + 1]])

    edge_list = np.zeros((A_data.size, 2))
    score = np.zeros(A_data.size)
    edge_id = 0
    for i in range(N):
        neighbors = A_indices[A_indptr[i] : A_indptr[i + 1]]
        for j in neighbors:
            if i < j:
                edge_list[edge_id, 0] = i
                edge_list[edge_id, 1] = j
                score[edge_id] = deg[i] * deg[j]
                edge_id += 1
    edge_order = np.argsort(-score)
    edge_num = edge_id

    q = _calculateSurprise_(A_indptr, A_indices, A_data, N, x)

    for itnum in range(5):
        for ind in range(edge_num):
            edge_id = edge_order[ind]
            u = int(edge_list[edge_id, 0])
            v = int(edge_list[edge_id, 1])
            # Move node u to the other group and
            # compute the changes in the number of edges of different types
            newx = 1 - x[u]
            x[u] = newx

            # Propose a move
            qp = _calculateSurprise_(A_indptr, A_indices, A_data, N, x)

            if q < qp:  # If this is a bad move, then bring back the original
                x[u] = 1 - x[u]
            else:
                q = qp

            if np.random.rand() > 0.5:
                # move 1 to 0 for n=3 nodes
                nnz = np.nonzero(x)[0]
                if nnz.shape[0] == 0:
                    continue
                n = np.minimum(3, nnz.shape[0])
                flip = np.random.choice(nnz, n)
                x[flip] = 1 - x[flip]
            else:
                # move 0 to 1 for n=3 nodes
                nnz = np.nonzero(1 - x)[0]
                if nnz.shape[0] == 0:
                    continue
                n = np.minimum(3, nnz.shape[0])
                flip = np.random.choice(nnz, n)
                x[flip] = 1 - x[flip]

            qp = _calculateSurprise_(A_indptr, A_indices, A_data, N, x)
            if q < qp:
                x[flip] = 1 - x[flip]
            else:
                q = qp

    # Flip group index if group 1 is sparser than group 0
    Nc = np.sum(x)
    Np = N - Nc
    Ax = _mat_prod_(A_indptr, A_indices, A_data, N, x)
    Ax_minus = _mat_prod_(A_indptr, A_indices, A_data, N, 1 - x)
    Ecc = np.sum(np.multiply(Ax, x))
    Epp = np.sum(np.multiply(Ax_minus, (1 - x)))
    if Ecc * (Np * (Np - 1)) < Epp * (Nc * (Nc - 1)):
        x = 1 - x
    return x, q


@numba.jit(nopython=True, cache=True)
def _calculateSurprise_(A_indptr, A_indices, A_data, N, x):

    Nc = np.sum(x)
    Np = N - Nc

    if (Nc < 2) | (Np < 2) | (Nc > N - 2) | (Np > N - 2):
        return 0

    L = np.sum(A_data) / 2
    V = N * (N - 1) / 2
    Vc = Nc * (Nc - 1) / 2
    Vcp = Nc * Np

    # Ax = A * x
    Ax = _mat_prod_(A_indptr, A_indices, A_data, N, x)

    lc = np.sum(np.multiply(Ax, x)) / 2
    lcp = np.sum(np.multiply(Ax, 1 - x))
    lp = L - lc - lcp

    p = L / (N * (N - 1) / 2)
    pc = lc / (Nc * (Nc - 1) / 2)
    pp = lp / (Np * (Np - 1) / 2)
    pcp = lcp / (Nc * Np)

    S = (
        _slog(p, pp, 2 * L)
        + _slog((1 - p), (1 - pp), 2 * V - 2 * L)
        + _slog(pp, pc, 2 * lc)
        + _slog((1 - pp), (1 - pc), 2 * Vc - 2 * lc)
        + _slog(pp, pcp, 2 * lcp)
        + _slog((1 - pp), (1 - pcp), 2 * Vcp - 2 * lcp)
    )

    return S


@numba.jit(nopython=True, cache=True)
def _slog(numer, denom, s):
    if (s == 0) | (numer < 0) | (denom < 0):
        return 0
    denom = denom * (1.0 - 1e-10) + 1e-10
    numer = numer * (1.0 - 1e-10) + 1e-10
    v = s * np.log(numer / denom)
    return v


def _score_(A_indptr, A_indices, A_data, N, x):
    return -_calculateSurprise_(A_indptr, A_indices, A_data, N, x)
