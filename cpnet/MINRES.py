from .CPAlgorithm import *
from . import utils
import numba


@numba.jit(nopython=True, cache=True)
def _score_(A_indptr, A_indices, A_data, num_nodes, x):
    Q = 0.0
    mcc = 0
    mpp = 0
    ncc = 0
    for i in range(num_nodes):
        neighbors = A_indices[A_indptr[i] : A_indptr[i + 1]]
        for k, nei in enumerate(neighbors):
            mcc += x[i] * x[nei]
            mpp += (1 - x[nei]) * (1 - x[nei])
        ncc += x[i]

    Q = (ncc * ncc - mcc) + mpp
    Q = -Q

    return Q


class MINRES(CPAlgorithm):
    """MINRES algorithm.

    MINRES algorithm for finding discrete core-periphery pairs [1], [2].

    Examples
    --------
    Create this object.

    >>> import cpnet
    >>> mrs = cpnet.MINRES()

    **Core-periphery detection**

    Detect core-periphery structure in network G (i.e., NetworkX object):

    >>> mrs.detect(G)

    Retrieve the ids of the core-periphery pair to which each node belongs:

    >>> pair_id = mrs.get_pair_id()

    Retrieve the coreness:

    >>> coreness = mrs.get_coreness()

    .. note::

       This algorithm accepts unweighted and undirected networks only.
       Also, the algorithm assigns all nodes into the same core-periphery pair by construction, i.e., c[node_name] =0 for all node_name.
       This algorithm is deterministic, i.e, one obtains the same result at each run.

    .. rubric:: References

    [1] J. P. Boyd, W. J Fitzgerald, M. C. Mahutga, and D. A. Smith. Computing continuous core/periphery structures for social relations data with MINRES/SVD. Soc.~Netw., 32:125–137, 2010.

    [2] S. Z. W.~ Lip. A fast algorithm for the discrete core/periphery bipartitioning problem. arXiv, pages 1102.5511, 2011.

    """

    def __init__(self):
        self.num_runs = 0

    def detect(self, G):
        """Detect a single core-periphery pair using the MINRES algorithm.

        Parameters
        ----------
        G : NetworkX graph object

        Examples
        --------

        >>> import networkx as nx
        >>> import cpnet
        >>> G = nx.karate_club_graph()  # load the karate club network.
        >>> mrs = cp.MINRES()
        >>> mrs.detect(G)



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
        return [_score_(A.indptr, A.indices, A.data, A.shape[0], x)]
