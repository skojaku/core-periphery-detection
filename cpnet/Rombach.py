from .CPAlgorithm import *
from simanneal import Annealer
import random
from . import utils
import numba


class Rombach(CPAlgorithm):
    """Rombach's algorithm for finding continuous core-periphery structure.

    Parameters
    ----------
    num_runs : int
        Number of runs of the algorithm  (optional, default: 1).

    alpha : float
        Sharpness of core-periphery boundary (optional, default: 0.5).

        alpha=0 or alpha=1 gives the fuzziest or sharpest boundary, respectively.

    beta : float
        Fraction of peripheral nodes (optional, default: 0.8)

    algorithm : str
        Optimisation algorithm (optional, default: 'ls')
            In the original paper [1], the authors adopted a simulated annealing to optimise the objective function, which is computationally demanding.
            To mitigate the computational cost, a label switching algorithm is implemented in cpnet.
            One can choose either algorithm by specifying algorithm='ls' (i.e., label switching) or algorithm='sa' (i.e., simulated annealing).

    .. note::

       The parameters of the simulated annealing such as the initial temperature and cooling schedule are different from those used in the original paper [1].


    Examples
    --------
    Create this object.

    >>> import cpnet
    >>> rb = cpnet.Rombach()

    **Core-periphery detection**

    Detect core-periphery structure in network G (i.e., NetworkX object):

    >>> rb.detect(G)

    Retrieve the ids of the core-periphery pair to which each node belongs:

    >>> pair_id = rb.get_pair_id()

    Retrieve the coreness:

    >>> coreness = rb.get_coreness()

    .. note::

       This algorithm can accept unweighted and weighted networks.
       The algorithm assigns all nodes into the same core-periphery pair by construction, i.e., c[node_name] =0 for all node_name.

    .. rubric:: Reference

        [1] P. Rombach, M. A. Porter, J. H. Fowler, and P. J. Mucha. Core-Periphery Structure in Networks (Revisited). SIAM Review, 59(3):619–646, 2017

    """

    def __init__(self, num_runs=10, alpha=0.5, beta=0.8, algorithm="ls"):
        self.num_runs = num_runs
        self.alpha = alpha
        self.beta = beta
        self.algorithm = algorithm

    def detect(self, G):
        """Detect a single core-periphery pair.

        Parameters
        ----------
        G : NetworkX graph object

        Examples
        --------
        >>> import networkx as nx
        >>> import cpnet
        >>> G = nx.karate_club_graph()  # load the karate club network.
        >>> rb = cp.Rombach(algorithm='ls') # label switching algorithm
        >>> rb.detect(G)
        >>> rb = cp.Rombach(algorithm='sa') # simulated annealing
        >>> rb.detect(G)

        """

        Qbest = -100
        cbest = 0
        xbest = 0
        qbest = 0
        A, nodelabel = utils.to_adjacency_matrix(G)
        self.Q_ = 0
        for i in range(self.num_runs):
            if self.algorithm == "ls":
                x, Q = self._label_switching(A, self.alpha, self.beta)
            elif self.algorithm == "sa":
                x, Q = self._simaneal(A, nodelist, self.alpha, self.beta)
            if Qbest < Q:
                Qbest = Q
                xbest = x
                qbest = Q

        self.nodelabel = nodelabel
        self.c_ = np.zeros(x.size).astype(int)
        self.x_ = x.astype(int)
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
        """Swaps two nodes"""
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
