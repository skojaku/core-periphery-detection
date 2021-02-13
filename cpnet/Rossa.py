import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

from . import utils
from .CPAlgorithm import CPAlgorithm


class Rossa(CPAlgorithm):
    """Rossa's algorithm for finding continuous core-periphery structure.

    This algorithm finds multiple core-periphery pairs in networks.

    F. Rossa, F. Dercole, and C. Piccardi. Profiling core-periphery network structure by random walkers. Scientific Reports, 3, 1467, 2013

    .. highlight:: python
    .. code-block:: python

        >>> import cpnet
        >>> alg = cpnet.Rossa()
        >>> alg.detect(G)
        >>> pair_id = alg.get_pair_id()
        >>> coreness = alg.get_coreness()

    .. note::

        - [x] weighted
        - [x] directed
        - [ ] multiple groups of core-periphery pairs
        - [ ] continuous core-periphery structure
    """

    def __init__(self):
        pass

    def detect(self, G):
        """Detect core-periphery structure.

        :param G: Graph
        :type G: networkx.Graph or scipy sparse matrix
        :return: None
        :rtype: None
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
        return [1 - np.mean(x)]

    def _detect(self, A):

        N = A.shape[0]
        deg = np.array(A.sum(axis=0)).reshape(-1)
        x = np.zeros((N, 1))

        idx = self._argmin2(deg)

        x[idx] = 1
        ak = deg[idx]
        bk = 0
        alpha = np.zeros(N)

        for _k in range(1, N):

            denom = np.maximum(1, np.max(ak * (ak + deg)))
            # denom = np.asscalar(np.max([1, np.max(ak * (ak + deg))]))
            score = (2 * ak * (x.T * A) - bk * deg) / denom

            score[x.T > 0] = np.Infinity
            score = np.squeeze(np.asarray(score))
            idx = self._argmin2(score)
            x[idx] = 1
            ak = ak + deg[idx]
            bk = np.dot(x.T * A, x)[0, 0]

            alpha[idx] = bk / max(1, ak)

        return np.zeros(N).astype(int), alpha

    def _argmin2(self, b):
        return np.random.choice(np.flatnonzero(b == b.min()))
