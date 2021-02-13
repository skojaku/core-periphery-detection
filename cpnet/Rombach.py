import random

import numba
import numpy as np
from simanneal import Annealer

from . import utils
from .CPAlgorithm import CPAlgorithm


class Rombach(CPAlgorithm):
    """Rombach's algorithm for finding continuous core-periphery structure.

    P. Rombach, M. A. Porter, J. H. Fowler, and P. J. Mucha. Core-Periphery Structure in Networks (Revisited). SIAM Review, 59(3):619–646, 2017

    .. highlight:: python
    .. code-block:: python

        >>> import cpnet
        >>> alg = cpnet.Rombach()
        >>> alg.detect(G)
        >>> pair_id = alg.get_pair_id()
        >>> coreness = alg.get_coreness()

    .. note::

        - [x] weighted
        - [ ] directed
        - [ ] multiple groups of core-periphery pairs
        - [x] continuous core-periphery structure
    """

    def __init__(self, num_runs=10, alpha=0.5, beta=0.8, algorithm="ls"):
        """[summary]

        :param num_runs: Number of runs of the algorithm, defaults to 10
        :type num_runs: int, optional
        :param alpha: Sharpness of core-periphery boundary, defaults to 0.5
        :type alpha: float, optional
        :param beta: Fraction of peripheral nodes, defaults to 0.8
        :type beta: float, optional
        :param algorithm: Optimisation algorithm, defaults to "ls"
        :type algorithm: str, optional

        .. note::

            In the original paper, the authors adopted a simulated annealing to optimise the objective function, which is computationally demanding.
            To mitigate the computational cost, a label switching algorithm is implemented in cpnet.
            One can choose either algorithm by specifying algorithm='ls' (i.e., label switching) or algorithm='sa' (i.e., simulated annealing).
        """
        self.num_runs = num_runs
        self.alpha = alpha
        self.beta = beta
        self.algorithm = algorithm

    def detect(self, G):
        """Detect core-periphery structure.

        :param G: Graph
        :type G: networkx.Graph or scipy sparse matrix
        :return: None
        :rtype: None
        """

        Qbest = -100
        xbest = 0
        qbest = 0
        A, nodelabel = utils.to_adjacency_matrix(G)
        self.Q_ = 0
        for _i in range(self.num_runs):
            if self.algorithm == "ls":
                x, Q = self._label_switching(A, self.alpha, self.beta)
            elif self.algorithm == "sa":
                x, Q = self._simaneal(A, nodelist, self.alpha, self.beta)
            if Qbest < Q:
                Qbest = Q
                xbest = x
                qbest = Q

        self.nodelabel = nodelabel
        self.c_ = np.zeros(xbest.size).astype(int)
        self.x_ = xbest
        self.Q_ = qbest
        self.qs_ = [qbest]

    def _label_switching(self, A, alpha, beta):

        ndord = _rombach_label_switching_(
            A.indptr, A.indices, A.data, A.shape[0], self.alpha, self.beta
        )
        x = np.array(
            [_calc_coreness(order, A.shape[0], alpha, beta) for order in ndord]
        )
        Q = x.T @ A @ x
        return x, Q

    def _simaneal(self, A, nodelist, alpha, beta):

        N = A.shape[0]

        nodes = list(range(N))
        random.shuffle(nodes)
        nodes = np.array(nodes)

        sa = SimAlg(A, nodes, self.alpha, self.beta)
        od, self.Q_ = sa.anneal()

        x = sa.corevector(od, self.alpha, self.beta)
        x = x.T.tolist()[0]

        Q = x.T @ A @ x
        return x, Q

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
        return [x.T @ A @ x]


class SimAlg(Annealer):
    def __init__(self, A, x, alpha, beta):

        self.state = x
        self.A = A
        self.alpha = alpha
        self.beta = beta
        self.Tmax = 1
        self.Tmin = 1e-8
        self.steps = 10000
        self.updates = 100

    def move(self):
        """Swaps two nodes."""
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[[a, b]] = self.state[[b, a]]

    def energy(self):
        return self.eval(self.state)

    def eval(self, od):

        x = self.corevector(od, self.alpha, self.beta)

        return -np.asscalar(np.dot(x.T * self.A, x)[0, 0])

    def corevector(self, x, alpha, beta):
        N = len(x)
        bn = np.floor(beta * N)
        cx = (x <= bn).astype(int)
        px = (x > bn).astype(int)

        c = (1.0 - alpha) / (2.0 * bn) * x * cx + (
            (x * px - bn) * (1.0 - alpha) / (2.0 * (N - bn)) + (1.0 + alpha) / 2.0
        ) * px
        return np.asmatrix(np.reshape(c, (N, 1)))


@numba.jit(nopython=True, cache=True)
def _calc_coreness(order, N, alpha, beta):
    c = 0.0
    bn = np.floor(N * beta)
    if order <= bn:
        c = (1.0 - alpha) / (2.0 * bn) * order
    else:
        c = (order - bn) * (1.0 - alpha) / (2 * (N - bn)) + (1 + alpha) / 2.0
    return c


@numba.jit(nopython=True, cache=True)
def _rowSum_score(A_indptr, A_indices, A_data, num_nodes, ndord, nid, sid, alpha, beta):
    retval = 0
    neighbors = A_indices[A_indptr[nid] : A_indptr[nid + 1]]
    weight = A_data[A_indptr[nid] : A_indptr[nid + 1]]
    for k, node_id in enumerate(neighbors):
        if node_id == sid:
            continue
        retval += weight[k] * _calc_coreness(ndord[node_id], num_nodes, alpha, beta)
        # bn = np.floor(num_nodes * beta)
        # if ndord[node_id] <= bn:
        #    c = (1.0-alpha) / (2.0*bn) * ndord[node_id]
        # else:
        #    c = (ndord[node_id]-bn)*(1.0-alpha)/(2*(num_nodes-bn))  + (1+alpha)/2.0
        # retval+=weight[k] * c
    return retval


@numba.jit(nopython=True, cache=True)
def _calc_swap_gain(
    A_indptr, A_indices, A_data, num_nodes, ndord, nid, sid, alpha, beta
):
    c_nid = _calc_coreness(ndord[nid], num_nodes, alpha, beta)
    c_sid = _calc_coreness(ndord[sid], num_nodes, alpha, beta)
    rowSum_nid = _rowSum_score(
        A_indptr, A_indices, A_data, num_nodes, ndord, nid, sid, alpha, beta
    )
    rowSum_sid = _rowSum_score(
        A_indptr, A_indices, A_data, num_nodes, ndord, sid, nid, alpha, beta
    )
    dQ = (c_sid - c_nid) * rowSum_nid + (c_nid - c_sid) * rowSum_sid
    return dQ


@numba.jit(nopython=True, cache=True)
def _rombach_label_switching_(A_indptr, A_indices, A_data, N, alpha, beta):
    ndord = np.arange(N)
    order = ndord.copy()
    isupdated = False
    itmax = 100
    itnum = 0
    while (isupdated is False) and (itnum < itmax):
        isupdated = False
        np.random.shuffle(order)
        for i in range(N):
            nid = order[i]  # Get the id of node we shall update
            nextnid = nid
            dQmax = -N
            for sid in range(N):
                if sid == nid:
                    continue
                # calc swap gain
                dQ = _calc_swap_gain(
                    A_indptr, A_indices, A_data, N, ndord, nid, sid, alpha, beta
                )
                if dQmax < dQ:
                    nextnid = sid
                    dQmax = dQ
            if dQmax <= 1e-10:
                continue
            isupdated = True
            tmp = ndord[nid]
            ndord[nid] = ndord[nextnid]
            ndord[nextnid] = tmp
        itnum += 1
    return ndord
