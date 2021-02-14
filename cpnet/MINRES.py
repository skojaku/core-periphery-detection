import numba
import numpy as np
from joblib import Parallel, delayed

from . import utils
from .adam import ADAM
from .CPAlgorithm import CPAlgorithm


class MINRES(CPAlgorithm):
    """MINRES algorithm.

    Boyd, J. P., Fitzgerald, W. J., Mahutga, M. C., & Smith, D. A. (2010).
    Computing continuous core/periphery structures for social relations data with MINRES/SVD.
    Social Networks, 32(2), 125–137. https://doi.org/10.1016/j.socnet.2009.09.003

    .. highlight:: python
    .. code-block:: python

        >>> import cpnet
        >>> alg = cpnet.MINRES()
        >>> alg.detect(G)
        >>> pair_id = alg.get_pair_id()
        >>> coreness = alg.get_coreness()

    .. note::

        - [x] weighted
        - [ ] directed
        - [ ] multiple groups of core-periphery pairs
        - [x] continuous core-periphery structure
    """

    def __init__(self, num_runs=10):
        """Initialize algorithm.

        :param num_runs: number of runs, defaults to 10
        :type num_runs: int, optional
        """
        self.num_runs = num_runs
        self.n_jobs = 1

    def detect(self, G):
        """Detect core-periphery structure.

        :param G: Graph
        :type G: networkx.Graph or scipy sparse matrix
        :return: None
        :rtype: None
        """

        A, nodelabel = utils.to_adjacency_matrix(G)

        def _detect(A, maxIt=10000):
            w = np.random.rand(A.shape[0])
            adam = ADAM()
            for _it in range(maxIt):
                wnorm = np.linalg.norm(w)

                grad = A @ w - (wnorm ** 2 - w ** 2) * w
                wnew = adam.update(w, grad, 0, False)

                diff = np.linalg.norm(wnew - w) / wnorm
                w = wnew.copy()
                if diff < 1e-2:
                    break

            Q = self._score(A, None, w)
            cids = np.zeros(A.shape[0])
            return {"cids": cids, "x": w, "q": Q[0]}

        res = Parallel(n_jobs=self.n_jobs)(
            delayed(_detect)(A) for i in range(self.num_runs)
        )
        res = max(res, key=lambda x: x["q"])
        cids, x, Q = res["cids"], res["x"], res["q"]
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
        Asq = np.sum(np.power(A.data, 2))
        wnorm = np.linalg.norm(x)
        Q = (
            Asq
            - 2 * x.reshape((1, -1)) @ A @ x.reshape((-1, 1))
            + wnorm * wnorm * (wnorm * wnorm - 1)
        )
        Q = Q[0][0]
        return [Q]
