import numpy as np
import itertools
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import metis
from tqdm import tqdm

from scipy.sparse import csr_matrix, isspmatrix

from collections import namedtuple

Points = namedtuple('Points', ['xyz', 'attr'])

def METIS_Partitioning(points, n_partitions = 4, n_neighbors = 128):
    print('METIS Partitioning...')

    Adj = kneighbors_graph(X = points, n_neighbors = n_neighbors, mode = 'connectivity', n_jobs = 10).toarray()

    G = nx.from_numpy_matrix(Adj)
    G = metis.networkx_to_metis(G)
    edgecuts, parts = metis.part_graph(G, n_partitions)

    new_node_indices = []
    for i in range(0,n_partitions):
        new_node_indices.append(np.argwhere(np.array(parts) == i).ravel())

    return new_node_indices

def Spectral_Clustering_Partitioning(points, n_clusters = 4, n_components = 4, n_neighbors = 64):
    print('Spectral Clustering Partitioning...')

    clustering = SpectralClustering(n_clusters, n_components = n_components, assign_labels='discretize', random_state=0, affinity='nearest_neighbors', n_neighbors = n_neighbors).fit(points)

    new_node_indices = []
    for i in range(0,n_clusters):
        new_node_indices.append(np.argwhere(np.array(clustering.labels_) == i).ravel().tolist())

    return new_node_indices

def sortMatrixRows(A):
    ## Get the number of rows and columns in the matrix
    M = np.shape(A)
    N = M[0]
    M = M[1]

    ## Sort A
    A = A[A[:, M-1].argsort()]
    for column in reversed(range(0,M-2)):
        A = A[A[:, column].argsort(kind='mergesort')]

    A = np.flipud(A)

    return A

def findBlueNoiseSamplingLatticeFromSparcyMatrix (S, N, K):
    '''
    Returns a binary vector, Xs, with 1s in the nodes that are sampled (preserved) and 0s in the doscarded nodes.
    For input, the usser suplies an adjacency matrix with n-self loops (main diagonal all 0s) and positive, real-valued
    weights. The integer, N, is the number of nodes that are sampled.

    Inputs:
        S: Adjacency matrix in Sparse matrix format.
        N: Number of desired sampled nodes.
        K: Number of desired iterations.

    Output:
        Xs: Binary vector of size (len(S), 1) with N ones that represent the sampled nodes.
    '''

    record = []
    np.random.seed(1)
    

    ## Get the number of nodes in S from the number of rows in S
    if isspmatrix(S):
        SM = S
        Num_Nodes = S.shape[0]
    else:
        print('S is not an sparse matrix')
        return

    ## Generate initial sampling lattice
    if isinstance(N, int):
        Xs = np.zeros((Num_Nodes,1))
        indices = np.random.permutation(Num_Nodes)
        for i in range(0,N):
            Xs[indices[i]] = 1
    else:
        print('N is not a positive integer')
        return

    ## Perform K iterations

    for iteration in tqdm(range(0,K)):
        A = np.zeros((Num_Nodes,10))
        A[:,0] = np.transpose(Xs)
        for column in range(1,9):
            A[:,column] = SM * A[:,column - 1]
            A[:,column] = A[:,column]/max(A[:,column])
        A[:,-1] = np.array(range(0,Num_Nodes))
        A = sortMatrixRows(A)
        ## Swap the highest ranked node with the lowest ranked node
        Xs[int(A[0, -1])] = 0
        Xs[int(A[Num_Nodes-1, -1])] = 1

        record.append(Xs)
        columns = len(record)

        ## Convergence criteria doesn't work
        #if (columns>3) and (all(record[-1] == record[-3]) and all(record[-2] == record[-3])):
        #    return Xs
    return Xs

def Void_and_Cluster_Downsampling(points, n_clusters = 4, n_components = 4, n_neighbors = 64):
    
    ## Graph partitioning
    ## Create graph partitions using Spectral Clustering
    #new_node_indices = Spectral_Clustering_Partitioning(points_xyz.xyz, n_clusters, n_components, n_neighbors)

    # Create graph partitions using METIS
    new_node_indices = METIS_Partitioning(points.xyz, n_partitions=n_clusters)

    # Extract xyz coordinates for each partition using index list
    new_node_xyz = []
    new_attr = []
    for i in range(0, n_clusters):
        new_node_xyz.append(np.zeros((len(new_node_indices[i]), len(points.xyz[0]))))
        new_attr.append(np.zeros((len(new_node_indices[i]), len(points.attr[0]))))
        for j in range(0, len(new_node_indices[i])):
            new_node_xyz[i][j] = points.xyz[new_node_indices[i][j]]
            new_attr[i][j] = points.attr[new_node_indices[i][j]]

    new_A = []
    cluster_edges = []
    for i in range(0,n_clusters):
        new_A.append(kneighbors_graph(X = new_node_xyz[i], n_neighbors = 5, mode = 'connectivity', n_jobs = 10).toarray())
        cluster_edges.append(np.transpose(np.nonzero(new_A[i])))


    # Implement Void and Cluster here:
    print('Voiding and Clustering...')
    Sampling_list = []
    for i in range(0, n_clusters):
        S = csr_matrix(new_A[i])
        N = int(S.shape[0]*0.1)
        K = N
        print('Partition ', (i+1),':')
        Xs = findBlueNoiseSamplingLatticeFromSparcyMatrix (S, N, K)

        Sampling_list.append(Xs)

    # Sample points according to void and cluster list
    Sampled_PC = []
    sampled_xyz = []
    sampled_attr = []
    for i in range(0, n_clusters):
        Sampled_PC.append([])
        sampled_xyz.append([])
        sampled_attr.append([])
        for j in range(0, len(new_node_xyz[i])):
            try:
                if Sampling_list[i][j] == 0:
                    Sampled_PC[i].append(j)
                else:
                    pass
            except:
                pass

        sampled_xyz[i] = np.delete(new_node_xyz[i], Sampled_PC[i], 0).tolist()
        sampled_attr[i] = np.delete(new_attr[i], Sampled_PC[i], 0).tolist()

    new_xyz = list(itertools.chain.from_iterable(sampled_xyz))
    new_xyz = np.array(new_xyz)
    new_attr = list(itertools.chain.from_iterable(sampled_attr))
    new_attr = np.array(new_attr)

    # Build Point class
    Downsampled_points = Points(new_xyz, new_attr)

    return Downsampled_points
