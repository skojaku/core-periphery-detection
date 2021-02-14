import numba
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

from . import utils
from .CPAlgorithm import CPAlgorithm


class LowRankCore(CPAlgorithm):
    """LowRankCore algorithm.

    M. Cucuringu, P. Rombach, S. H. Lee, and M. A. Porter Detection of core-periphery structure in networks using spectral methods and geodesic paths. Euro. J. Appl. Math., 27:846–887, 2016.

    .. highlight:: python
    .. code-block:: python

    >>> import cpnet
    >>> lrc = cpnet.LowRankCore()
    >>> lrc.detect(G)
    >>> pair_id = lrc.get_pair_id()
    >>> coreness = lrc.get_coreness()

    .. note::

        - [ ] weighted
        - [ ] directed
        - [ ] multiple groups of core-periphery pairs
        - [ ] continuous core-periphery structure
    """

    def __init__(self, beta=0.1):
        """Initialize algorithm.

        :param beta: parameter of the algorithm. See the original paper., defaults to 0.1
        :type beta: float, optional
        """
        self.beta = beta

    def detect(self, G):
        """Detect a single core-periphery pair.

        :param G: Graph
        :type G: networkx.Graph or scipy sparse matrix
        """
        A, nodelabel = utils.to_adjacency_matrix(G)
        x = self._low_rank_core(A)

        Q = self._score(A, None, x)

        self.nodelabel = nodelabel
        self.c_ = np.zeros(A.shape[0]).astype(int)
        self.x_ = x.astype(int)
        self.Q_ = np.sum(Q)
        self.qs_ = Q

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
        N = A.shape[0]
        Mcc = np.dot(x.T @ A, x) / 2
        Mcp = np.dot(x.T @ A, (1 - x))
        Mpp = np.dot(x.T @ A, x) / 2
        i = np.sum(x)
        if i < 2 or i > N - 2:
            return [0.0]
        q = (
            Mcc / float(i * (i - 1) / 2)
            + Mcp / float(i * (N - i))
            - Mpp / float((N - i) * ((N - i) - 1) / 2)
        )
        return [q]

    def _find_cut(self, A, score, b):
        """Find the best cut that maximises the objective.

        :param A: adjacency matrix
        :type A: scipy sparse matrix
        :param score: score for each node
        :type score: numpy.ndarray
        :param b: prameter
        :type b: float
        :return: core vector
        :rtype: numpy.ndarray
        """

        N = A.shape[0]
        qc = np.zeros(N)
        qp = np.zeros(N)
        od = (-score).argsort()

        for i in range(b, N - b):
            x = np.zeros((N, 1))
            x[od[0:i]] = 1

            Mcc = np.dot(x.T @ A, x)[0, 0] / 2
            Mcp = np.dot(x.T @ A, (1 - x))[0, 0]
            Mpp = np.dot((1 - x).T * A, (1 - x))[0, 0] / 2
            qc[i] = (
                Mcc / float(i * (i - 1) / 2)
                + Mcp / float(i * (N - i))
                - Mpp / float((N - i) * ((N - i) - 1) / 2)
            )
            qp[i] = (
                Mcp / float(i * (N - i))
                + Mpp / float((N - i) * ((N - i) - 1) / 2)
                - Mcc / float(i * (i - 1) / 2)
            )

        idx_c = np.argmax(qc)
        idx_p = np.argmax(qp)

        if qc[idx_c] > qp[idx_p]:
            Q = qc[idx_c]
            x = np.zeros(N)
            x[od[0:idx_c]] = 1
        else:
            Q = qc[idx_p]
            x = np.ones(N)
            x[od[0:idx_p]] = 0

        Q = Q / N
        return x

    def _low_rank_core(self, A):
        """low rank core algorithm.

        :param A: adjacency matrix
        :type A: scipy sparse matrix
        :return: core vector
        :rtype: numpy.ndarray
        """

        N = A.shape[0]
        d, v = eigs(A, k=2, which="LM")

        At = (np.dot(v * diags(d), v.T) > 0.5).astype(int)
        score = At.sum(axis=0)

        x = self._find_cut(A, score, int(np.round(N * self.beta)))
        return x


class LapCore(CPAlgorithm):
    """LapCore algorithm.

    M. Cucuringu, P. Rombach, S. H. Lee, and M. A. Porter Detection of core-periphery structure in networks using spectral methods and geodesic paths. Euro. J. Appl. Math., 27:846–887, 2016.

    .. highlight:: python
    .. code-block:: python

    >>> import cpnet
    >>> lc = cpnet.LapCore()
    >>> lc.detect(G)
    >>> pair_id = lc.get_pair_id()
    >>> coreness = lc.get_coreness()

    .. note::

        - [ ] weighted
        - [ ] directed
        - [ ] multiple groups of core-periphery pairs
        - [ ] continuous core-periphery structure
    """

    def __init__(self, beta=0.1):
        self.beta = beta

    def detect(self, G):
        """Detect core-periphery structure.

        :param G: Graph
        :type G: networkx.Graph or scipy sparse matrix
        """

        A, nodelabel = utils.to_adjacency_matrix(G)
        x = self._lap_core(A)
        Q = self._score(A, None, x)

        self.nodelabel = nodelabel
        self.c_ = np.zeros(A.shape[0]).astype(int)
        self.x_ = x.astype(int)
        self.Q_ = np.sum(Q)
        self.qs_ = Q

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
        N = A.shape[0]
        Mcc = np.dot(x.T * A, x) / 2
        Mcp = np.dot(x.T * A, (1 - x))
        Mpp = np.dot(x.T * A, x) / 2
        i = np.sum(x)
        if i < 2 or i > N - 2:
            return [0.0]

        q = (
            Mcc / float(i * (i - 1) / 2)
            + Mcp / float(i * (N - i))
            - Mpp / float((N - i) * ((N - i) - 1) / 2)
        )
        return [q]

    def _find_cut(self, A, score, b):
        """Find the best cut that maximises the objective.

        :param A: adjacency matrix
        :type A: scipy sparse matrix
        :param score: score for each node
        :type score: numpy.ndarray
        :param b: prameter
        :type b: float
        :return: core vector
        :rtype: numpy.ndarray
        """
        N = A.shape[0]
        qc = np.zeros(N)
        qp = np.zeros(N)
        od = (-score).argsort()

        for i in range(b, N - b):
            x = np.zeros((N, 1))
            x[od[0:i]] = 1

            Mcc = np.dot(x.T * A, x)[0, 0] / 2
            Mcp = np.dot(x.T * A, (1 - x))[0, 0]
            Mpp = np.dot((1 - x).T * A, (1 - x))[0, 0] / 2
            qc[i] = (
                Mcc / float(i * (i - 1) / 2)
                + Mcp / float(i * (N - i))
                - Mpp / float((N - i) * ((N - i) - 1) / 2)
            )
            qp[i] = (
                Mcp / float(i * (N - i))
                + Mpp / float((N - i) * ((N - i) - 1) / 2)
                - Mcc / float(i * (i - 1) / 2)
            )

        idx_c = np.argmax(qc)
        idx_p = np.argmax(qp)

        if qc[idx_c] > qp[idx_p]:
            Q = qc[idx_c]
            x = np.zeros(N)
            x[od[0:idx_c]] = 1
        else:
            Q = qc[idx_p]
            x = np.ones(N)
            x[od[0:idx_p]] = 0

        Q = Q / N
        return x

    def _lap_core(self, A):
        """low rank core algorithm.

        :param A: adjacency matrix
        :type A: scipy sparse matrix
        :return: core vector
        :rtype: numpy.ndarray
        """
        N = A.shape[0]
        deg = np.array(A.sum(axis=1)).reshape(-1)
        denom = np.zeros(N)
        denom[deg > 0] = 1.0 / (deg[deg > 0] + 1.0)
        T = diags(denom) * A - diags(np.ones(N))
        d, v = eigs(T, k=1, which="SR")
        x = self._find_cut(A, v.T[0], int(np.round(N * self.beta)))
        return x


class LapSgnCore(CPAlgorithm):
    """LowSgnCore algorithm.

    M. Cucuringu, P. Rombach, S. H. Lee, and M. A. Porter Detection of core-periphery structure in networks using spectral methods and geodesic paths. Euro. J. Appl. Math., 27:846–887, 2016.

    .. highlight:: python
    .. code-block:: python

    >>> import cpnet
    >>> lsc = cpnet.LapSgnCore()
    >>> lsc.detect(G)
    >>> pair_id = lsc.get_pair_id()
    >>> coreness = lsc.get_coreness()

    .. note::

        - [ ] weighted
        - [ ] directed
        - [ ] multiple groups of core-periphery pairs
        - [ ] continuous core-periphery structure
    """

    def __init__(self, beta=0.1):
        """Initialize algorithm.

        :param beta: parameter of the algorithm. See the original paper., defaults to 0.1
        :type beta: float, optional
        """
        self.beta = beta

    def detect(self, G):
        """Detect a single core-periphery pair.

        :param G: Graph
        :type G: networkx.Graph or scipy sparse matrix
        """

        A, nodelabel = utils.to_adjacency_matrix(G)
        x = self._lapsgn_core(A)

        Q = self._score(A, None, x)

        self.nodelabel = nodelabel
        self.c_ = np.zeros(A.shape[0]).astype(int)
        self.x_ = x.astype(int)
        self.Q_ = np.sum(Q)
        self.qs_ = Q

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
        N = A.shape[0]
        Mcc = np.dot(x.T @ A, x) / 2
        Mcp = np.dot(x.T @ A, (1 - x))
        Mpp = np.dot(x.T @ A, x) / 2
        i = np.sum(x)
        if i < 2 or i > N - 2:
            return [0.0]

        q = (
            Mcc / float(i * (i - 1) / 2)
            + Mcp / float(i * (N - i))
            - Mpp / float((N - i) * ((N - i) - 1) / 2)
        )
        return [q]

    def _lapsgn_core(self, A):
        """lapsgn  algorithm.

        :param A: adjacency matrix
        :type A: scipy sparse matrix
        :return: core vector
        :rtype: numpy.ndarray
        """

        N = A.shape[0]
        deg = np.array(A.sum(axis=0)).reshape(-1)
        denom = np.zeros(N)
        denom[deg > 0] = 1.0 / (deg[deg > 0] + 1.0)
        T = diags(denom) * A - diags(np.ones(N))
        d, v = eigs(T, k=1, which="SR")
        v = np.sign(v)

        x = (v.T > 0).astype(float)
        x = np.array(x).reshape(-1)
        if self._score(A, None, x) < self._score(A, None, 1 - x):
            x = 1 - x
        return x
