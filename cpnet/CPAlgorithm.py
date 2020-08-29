from abc import ABCMeta, abstractmethod
import abc
import networkx as nx
import numpy as np
from . import utils


class CPAlgorithm(metaclass=ABCMeta):
    def __init__(self):
        self.x_ = []
        self.c_ = []
        self.Q_ = []
        self.qs_ = []

    @abstractmethod
    def detect(self):
        """ Private """
        pass

    @abstractmethod
    def _score(self, A, c, x):
        """ Private """
        pass

    def pairing(self, labels, a):
        return dict(zip(labels, a))

    def depairing(self, labels, d):
        return np.array([d[x] for x in labels])

    def get_pair_id(self):
        """Get core-periphery pair ID of each node.


        Returns
        -------
        c : dict
            Key: Node name
            Value: IDs of core-periphery pair to which it belongs.

        """
        return self.pairing(self.nodelabel, self.c_)

    def get_coreness(self):
        """Get the coreness of each node"""
        return self.pairing(self.nodelabel, self.x_)

    def score(self, G, c, x):
        """Get score of core-periphery pairs.

        Parameters
        ----------
        G : Graph object.
        c : Dict object,  the keys and values of which are the name of node and its ID of belonging core-periphery pair.


        Returns
        -------
        q : List. q[i] is the quality of core-periphery pair i.

        """
        A, nodelabel = utils.to_adjacency_matrix(G)
        c = self.depairing(nodelabel, c)
        x = self.depairing(nodelabel, x)
        return self._score(A, c, x)
