import numba
import numpy as np

from . import utils
from .CPAlgorithm import CPAlgorithm


@numba.jit(nopython=True, cache=True)
def _score_(A_indptr, A_indices, A_data, num_nodes, x):
    Q = 0.0
    mcc = 0
    mpp = 0
    ncc = 0
    for i in range(num_nodes):
        neighbors = A_indices[A_indptr[i] : A_indptr[i + 1]]
        for _k, nei in enumerate(neighbors):
            mcc += x[i] * x[nei]
            mpp += (1 - x[nei]) * (1 - x[nei])
        ncc += x[i]

    Q = (ncc * ncc - mcc) + mpp
    Q = -Q

    return Q


class Lip(CPAlgorithm):
    """Lip's algorithm.

    S. Z. W.~ Lip. A fast algorithm for the discrete core/periphery bipartitioning problem. arXiv, pages 1102.5511, 2011.

    .. highlight:: python
    .. code-block:: python

        >>> import cpnet
        >>> alg = cpnet.Lip()
        >>> alg.detect(G)
        >>> pair_id = alg.get_pair_id()
        >>> coreness = alg.get_coreness()

    .. note::

        - [ ] weighted
        - [ ] directed
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

        x = self._detect(np.array(A.sum(axis=1)).reshape(-1))
        cids = np.zeros(A.shape[0]).astype(int)

        Q = self._score(A, None, x)
        self.nodelabel = nodelabel
        self.c_ = cids
        self.x_ = x
        self.Q_ = Q
        self.qs_ = Q

    def _detect(self, deg):
        M = np.sum(deg)
        order = np.argsort(-deg)
        Z = M
        Zbest = np.inf
        kbest = 0
        for k in range(len(deg)):
            Z = Z + k - 1 - deg[order[k]]
            if Z < Zbest:
                kbest = k
                Zbest = Z
        _x = np.zeros(len(deg))
        _x[order[: kbest + 1]] = 1

        return _x

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
        return [_score_(A.indptr, A.indices, A.data, A.shape[0], x)]
