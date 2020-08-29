from .CPAlgorithm import *
from . import utils
import numba


class KM_ER(CPAlgorithm):
    """Kojaku-Masuda algorithm with the Erdos-Renyi random graph.
	
	This algorithm finds multiple core-periphery pairs in networks. 
	In the detection of core-periphery pairs, the Erdos-Renyi random graph is used as the null model. 	
	
	Parameters
	----------
	num_runs : int
		Number of runs of the algorithm  (optional, default: 1).  

	Examples
	--------
	Create this object.

	>>> import cpnet	
	>>> km = cpnet.KM_ER()
	
	**Core-periphery detection**
	
	Detect core-periphery structure in network G (i.e., NetworkX object):
	
	>>> km.detect(G) 
	
	Retrieve the ids of the core-periphery pair to which each node belongs:
	
	>>> pair_id = km.get_pair_id() 
	
	Retrieve the coreness:

	>>> coreness = km.get_coreness() 
		
	.. note::

	   This algorithm can accept unweighted and weighted networks.

	.. rubric:: Reference

        [1] S. Kojaku and N. Masuda. Finding multiple core-periphery pairs in network. Phys. Rev. 96(5):052313, 2017 

	"""

    def __init__(self, num_runs=10):
        self.num_runs = num_runs
        self.alpha = 0.5

    def detect(self, G):
        """Detect multiple core-periphery pairs.
	
		Parameters
		----------
		G : NetworkX graph object
		
		Examples
		--------
		>>> import networkx as nx
		>>> import cpnet
		>>> G = nx.karate_club_graph()  # load the karate club network. 
		>>> km = cp.KM_ER() # label switching algorithm
		>>> km.detect(G)
	"""

        A, nodelabel = utils.to_adjacency_matrix(G)

        c = x = None
        Q = -np.inf
        for i in range(self.num_runs):
            cidsi, xi = _label_switching_(
                A.indptr, A.indices, A.data, A.shape[0], alpha=self.alpha
            )
            _, cidsi = np.unique(cidsi, return_inverse=True)

            Qi, qsi = _score_(
                A.indptr, A.indices, A.data, cidsi, xi, A.shape[0], alpha=self.alpha
            )

            if Qi > Q:
                Q = Qi
                qs = qsi.copy()
                cids = cidsi.copy()
                x = xi.copy()

        self.nodelabel = nodelabel
        self.c_ = cids.astype(int)
        self.x_ = x.astype(int)
        self.Q_ = Q
        self.qs_ = qs

    def _score(self, A, c, x):
        num_nodes = A.shape[0]
        Q, qs = _score_(A.indptr, A.indices, A.data, c, x, num_nodes, self.alpha)
        return qs

    def significance(self):
        return self.pvalues


@numba.jit(nopython=True, cache=True)
def _label_switching_(A_indptr, A_indices, A_data, num_nodes, alpha=0.5, itnum_max=50):

    x = np.ones(num_nodes)
    deg = np.zeros(num_nodes)
    Nc = np.zeros(num_nodes)
    Np = np.zeros(num_nodes)
    cids = np.arange(num_nodes)
    for nid in range(num_nodes):
        deg[nid] = np.sum(A_data[A_indptr[nid] : A_indptr[nid + 1]])
        Nc[cids[nid]] += x[nid]
        Np[cids[nid]] += 1 - x[nid]

    M = np.sum(deg) / 2
    rho = M / (num_nodes * (num_nodes - 1) / 2)
    for it in range(itnum_max):
        order = np.random.choice(num_nodes, size=num_nodes, replace=False)
        updated_node_num = 0

        for k, node_id in enumerate(order):

            # Get the weight and normalized weight
            neighbors = A_indices[A_indptr[node_id] : A_indptr[node_id + 1]]
            weight = A_data[A_indptr[node_id] : A_indptr[node_id + 1]]

            # Calculate the grain
            clist = np.unique(cids[neighbors])
            next_cid = -1
            qself = 0
            dqmax = 0
            for cprime in clist:
                for xprime in [0, 1]:
                    neis = neighbors[cids[neighbors] == cprime]
                    non_pp_edges = x[neis] + (1 - x[neis]) * xprime
                    dq = (1 - alpha) * np.sum(
                        weight[cids[neighbors] == cprime] * non_pp_edges
                    )

                    Nc_prime = Nc[cprime] - xprime * (cprime == cids[node_id])
                    Np_prime = Np[cprime] - (1 - xprime) * (cprime == cids[node_id])
                    dq -= alpha * rho * (Nc_prime + Np_prime * xprime)

                    if (cprime == cids[node_id]) and (xprime == x[node_id]):
                        qself = dq
                        continue

                    if dqmax < dq:
                        next_cid = cprime
                        next_x = xprime
                        dqmax = dq

            dqmax -= qself
            if dqmax <= 1e-16:
                continue

            Nc[cids[node_id]] -= x[node_id]
            Np[cids[node_id]] -= 1 - x[node_id]
            Nc[next_cid] += next_x
            Np[next_cid] += 1 - next_x

            cids[node_id] = next_cid
            x[node_id] = next_x

            updated_node_num += 1

        if (updated_node_num / num_nodes) < 1e-3:
            break

    return cids, x


@numba.jit(nopython=True, cache=True)
def _score_(A_indptr, A_indices, A_data, _c, _x, num_nodes, alpha=0.5):
    cids = np.unique(_c)
    K = int(np.max(cids) + 1)
    q = np.zeros(K)
    Nc = np.zeros(num_nodes)
    Np = np.zeros(num_nodes)
    rho = 0
    doubleM = 0
    for i in range(num_nodes):
        neighbors = A_indices[A_indptr[i] : A_indptr[i + 1]]
        weight = A_data[A_indptr[i] : A_indptr[i + 1]]
        for j, nei in enumerate(neighbors):
            if _c[nei] == _c[i]:
                q[_c[nei]] += weight[j] * (_x[i] + _x[nei] - _x[i] * _x[nei])
        Nc[_c[i]] += _x[i]
        Np[_c[i]] += 1 - _x[i]
        doubleM += np.sum(weight)

    rho = doubleM / (num_nodes * num_nodes)
    Q = 0
    for k in range(K):
        q[k] = (1 - alpha) * q[k] - alpha * rho * (Nc[k] * Nc[k] + 2 * Nc[k] * Np[k])
        q[k] /= doubleM
        Q += q[k]

    return Q, q
