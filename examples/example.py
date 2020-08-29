"""Detect core-periphery structure in empirical networks.

"""
import cpalgorithm as cp
import scipy.stats as stats
import networkx as nx
import cpalgorithm as cpa
import numpy as np 
import sys

# load graph 
G = nx.karate_club_graph()
G = G.to_undirected()

# load algorithm
alg = cpa.BE()

# Core-periphery detection
alg.detect(G)

print("core-periphery IDs")
print(alg.get_pair_id())

print("Coreness ")
print(alg.get_coreness())
