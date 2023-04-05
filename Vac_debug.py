import numpy as np

import Void_and_Cluster_Sampling

from scipy.sparse import csr_matrix, isspmatrix

import networkx

G = networkx.path_graph(10)
A = networkx.adjacency_matrix(G)

Xs = Void_and_Cluster_Sampling.findBlueNoiseSamplingLatticeFromSparcyMatrix(A,3,3)

print(Xs)