from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from . import utils
from .CPAlgorithm import *


class Rossa(CPAlgorithm):
    """Rossa's algorithm for finding continuous core-periphery structure.

    Examples
    --------
    Create this object.

    >>> import cpnet
    >>> rs = cpnet.Rossa()

    **Core-periphery detection**

    Detect core-periphery structure in network G (i.e., NetworkX object):

    >>> rs.detect(G)

    Retrieve the ids of the core-periphery pair to which each node belongs:

    >>> pair_id = rs.get_pair_id()

    Retrieve the coreness:

    >>> coreness = rs.get_coreness()

    .. note::

       This algorithm can accept unweighted and weighted networks.
       The algorithm assigns all nodes into the same core-periphery pair by construction, i.e., c[node_name] =0 for all node_name.


    .. rubric:: Reference

    F. Rossa, F. Dercole, and C. Piccardi. Profiling core-periphery network structure by random walkers. Scientific Reports, 3, 1467, 2013


    """

    def __init__(self):
        return

    def detect(self, G):
        """Detect a single core-periphery structure.

        Parameters
        ----------
        G : NetworkX graph object

        Examples
        --------
        >>> import networkx as nx
        >>> import cpnet
        >>> G = nx.karate_club_graph()  # load the karate club network.
        >>> rs = cp.Rossa()
        >>> rs.detect(G)
        """
        A, nodelabel = utils.to_adjacency_matrix(G)

        cids, x = self._detect(A)
        Q = self._score(A, cids, x)[0]

        self.nodelabel = nodelabel
        self.c_ = cids.astype(int)
        self.x_ = x
        self.Q_ = Q
        self.qs_ = [Q]

    def _score(self, A, c, x):
        return [1 - np.mean(x)]

    def _detect(self, A):

        N = A.shape[0]
        deg = np.array(A.sum(axis=0)).reshape(-1)
        M = sum(deg) / 2.0
        x = np.zeros((N, 1))

        idx = self._argmin2(deg)

        x[idx] = 1
        ak = deg[idx]
        bk = 0
        alpha = np.zeros(N)

        for k in range(1, N):

            denom = np.asscalar(np.max([1, np.max(ak * (ak + deg))]))
            score = (2 * ak * (x.T * A) - bk * deg) / denom

            score[x.T > 0] = np.Infinity
            score = np.squeeze(np.asarray(score))
            idx = self._argmin2(score)
            x[idx] = 1
            ak = ak + deg[idx]
            bk = np.asscalar(np.dot(x.T * A, x)[0, 0])

            alpha[idx] = bk / max(1, ak)

        return np.zeros(N).astype(int), alpha

    def _argmin2(self, b):
        return np.random.choice(np.flatnonzero(b == b.min()))
