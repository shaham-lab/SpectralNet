import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors



def get_number_of_clusters(X: torch.Tensor,  n_samples: int, threshold: float) -> int:
    """
    Computes the number of clusters in the given dataset

    Args:
        X:          dataset
        n_samples:  number of samples to use for computing the number of clusters
        threshold:  threshold for the eigenvalues of the laplacian matrix. This 
                    threshold is used in order to find when the difference between 
                    the eigenvalues becomes large. 

    Returns:
        Number of clusters in the dataset
    """
    indices = torch.randperm(X.shape[0])[:n_samples]
    X = X[indices]
    
    W = get_affinity_matrix(X)
    L = get_laplacian(W)
    vals = get_eigenvalues(L)
    diffs = np.diff(vals)
    cutoff = np.argmax(diffs > threshold)
    num_clusters = cutoff + 1
    return num_clusters

def build_ann(X: torch.Tensor):
    """
    Builds approximate-nearest-neighbors object 
    that can be used to calculate the knn of a data-point

    Args:
        X:  dataset
    """
    X = X.view(X.size(0), -1)
    t = AnnoyIndex(X[0].shape[0], 'euclidean')
    for i, x_i in enumerate(X):
        t.add_item(i, x_i)

    t.build(50)
    t.save('ann_index.ann')


def make_batch_for_sparse_grapsh(batch_x: torch.Tensor) -> torch.Tensor:
    """
    Computes new batch of data points from the given batch (batch_x) 
    in case that the graph-laplacian obtained from the given batch is sparse.
    The new batch is computed based on the nearest neighbors of 0.25
    of the given batch

    Args:
        batch_x:    Batch of data points

    Returns:
        New batch of data points
    """

    batch_size = batch_x.shape[0]
    batch_size //= 5
    new_batch_x = batch_x[:batch_size]
    batch_x = new_batch_x
    n_neighbors = 5

    u = AnnoyIndex(batch_x[0].shape[0], 'euclidean')
    u.load('ann_index.ann')
    for x in batch_x:
        x = x.detach().cpu().numpy()
        nn_indices = u.get_nns_by_vector(x, n_neighbors)
        nn_tensors = [u.get_item_vector(i) for i in nn_indices[1:]]
        nn_tensors = torch.tensor(nn_tensors)
        new_batch_x = torch.cat((new_batch_x, nn_tensors))

    return new_batch_x


def get_laplacian(W: torch.Tensor) -> np.ndarray:
    """
    Computes the unnormalized Laplacian matrix, given the affinity matrix W

    Args:
        W (torch.Tensor):   Affinity matrix
    
    Returns:
        Laplacian matrix
    """

    W = W.detach().cpu().numpy()
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L


def sort_laplacian(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sorts the columns and the rows of the laplacian by the true lablel in order
    to see whether the sorted laplacian is a block diagonal matrix

    Args:
        L:  Laplacian matrix
        y:  labels

    Returns:
        Sorted laplacian
    """

    i = np.argsort(y)
    L = L[i, :]
    L = L[:, i]
    return L


def sort_matrix_rows(A: np.ndarray , y: np.ndarray) -> np.ndarray:
    """
    Sorts the rows of a matrix by a given order y

    Args:
        A:  Numpy ndarray
        y:  True labels
    """

    i = np.argsort(y)
    A = A[i, :]
    return A


def get_eigenvalues(A: np.ndarray) -> np.ndarray:
    """
    Computes the eigenvalues of a given matrix A and sorts them in increasing order

    Args:
        A:  Numpy ndarray

    Returns:
        Sorted eigenvalues
    """

    _, vals, _ = np.linalg.svd(A)
    sorted_vals = vals[np.argsort(vals)]
    return sorted_vals


def get_eigenvectors(A: np.ndarray) -> np.ndarray:
    """
    Computes the eigenvectors of a given matrix A and sorts them by the eigenvalues
    Args:
        A:  Numpy ndarray

    Returns:
        Sorted eigenvectors
    """

    vecs, vals, _ = np.linalg.svd(A)
    vecs = vecs[:, np.argsort(vals)]
    return vecs

def plot_eigenvalues(vals: np.ndarray):
    """
    Plot the eigenvalues of the laplacian

    Args:
        vals:   Eigenvalues
    """

    rang = range(len(vals))
    plt.plot(rang, vals)
    plt.show()

def get_laplacian_eigenvectors(V: torch.Tensor, y: np.ndarray) -> np.ndarray:
    """
    Returns eigenvectors of the laplacian when the data is in increasing order by the true label.
    i.e., the rows of the eigenvectors matrix V are sorted by the true labels in increasing order.

    Args:
        V:  Eigenvectors matrix
        y:  True labels
    """

    V = sort_matrix_rows(V, y)
    rang = range(len(y))
    return V, rang

def plot_laplacian_eigenvectors(V: np.ndarray, y: np.ndarray):
    """
    Plot the eigenvectors of the laplacian when the data is in increasing order by the true label.
    i.e., the rows of the eigenvectors matrix V are sorted by the true labels in increasing order.

    Args:
        V:  Eigenvectors matrix
        y:  True labels
    """

    # sort the rows of V
    V = sort_matrix_rows(V, y)
    rang = range(len(y))
    plt.plot(rang, V)
    plt.show()
    return plt


def plot_sorted_laplacian(W: torch.Tensor, y: np.ndarray):
    """
    Plot the block diagonal matrix that is obtained from the sorted laplacian

    Args:
        W:  Affinity matrix
        y:  True labels
    """
    L = get_laplacian(W)
    L = sort_laplacian(L, y)
    plt.imshow(L, cmap='hot', norm=colors.LogNorm())
    plt.imshow(L, cmap='flag')
    plt.show()


def get_nearest_neighbors(X: torch.Tensor, Y: torch.Tensor = None, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the distances and the indices of the 
    k nearest neighbors of each data point

    Args:
        X:              Batch of data points
        Y (optional):   Defaults to None.
        k:              Number of nearest neighbors to calculate. Defaults to 3.

    Returns:
        Distances and indices of each datapoint
    """

    if Y is None:
        Y = X
    if len(X) < k:
        k = len(X)
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(Y)
    Dis, Ids = nbrs.kneighbors(X)
    return Dis, Ids


def get_grassman_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Computes the Grassmann distance between the subspaces spanned by the columns of A and B

    Args:
        A:  Numpy ndarray
        B:  Numpy ndarray
    """

    M = np.dot(np.transpose(A), B)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = 1 - np.square(s)
    grassmann = np.sum(s)
    return grassmann


def compute_scale(Dis: np.ndarray, k: int = 2, med: bool = True, is_local: bool = True) -> np.ndarray:
    """
    Computes the scale for the Gaussian similarity function

    Args:
        Dis:        Distances of the k nearest neighbors of each data point.
        k:          Number of nearest neighbors. Defaults to 2.
        med:        Scale calculation method. Can be calculated by the median distance
                    from a data point to its neighbors, or by the maximum distance. 
        is_local:   Local distance (different for each data point), or global distance. Defaults to local.

    Returns:
        scale (global or local)
    """

    if is_local:
        if not med:
            scale = np.max(Dis, axis=1)
        else:
            scale = np.median(Dis, axis=1)
    else:
        if not med:
            scale = np.max(Dis[:, k - 1])
        else:
            scale = np.median(Dis[:, k - 1])
    return scale


def get_gaussian_kernel(D: torch.Tensor, scale, Ids: np.ndarray, device: torch.device, is_local: bool = True) -> torch.Tensor:   
    """
    Computes the Gaussian similarity function 
    according to a given distance matrix D and a given scale

    Args:
        D:      Distance matrix 
        scale:  scale
        Ids:    Indices of the k nearest neighbors of each sample
        device: Defaults to torch.device("cpu")
        is_local:  Determines whether the given scale is global or local 

    Returns:
        Matrix W with Gaussian similarities
    """

    if not is_local:
        # global scale
        W = torch.exp(-torch.pow(D, 2) / (scale ** 2))
    else:
        # local scales
        W = torch.exp(-torch.pow(D, 2).to(device) / (torch.tensor(scale).float().to(device).clamp_min(1e-7) ** 2))
    if Ids is not None:
        n, k = Ids.shape
        mask = torch.zeros([n, n]).to(device=device)
        for i in range(len(Ids)):
            mask[i, Ids[i]] = 1
        W = W * mask
    sym_W = (W + torch.t(W)) / 2.
    return sym_W


def plot_data_by_assignmets(X, assignments: np.ndarray):
    """
    Plots the data with the assignments obtained from SpectralNet.
    Relevant only for 2D data

    Args:
        X:                      Data
        cluster_assignments:    Cluster assignments 
    """

    plt.scatter(X[:, 0], X[:, 1], c=assignments)
    plt.show()

def calculate_cost_matrix(C: np.ndarray , n_clusters: int) -> np.ndarray:
    """
    Calculates the cost matrix for the Munkres algorithm

    Args:
        C (np.ndarray):     Confusion matrix
        n_clusters (int):   Number of clusters

    Returns:
        np.ndarray:        Cost matrix
    """
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices: np.ndarray) -> np.ndarray:
    """
    Gets the cluster labels from their indices

    Args:
        indices (np.ndarray):  Indices of the clusters

    Returns:
        np.ndarray:   Cluster labels
    """

    num_clusters = len(indices)
    cluster_labels = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def write_assignmets_to_file(assignments: np.ndarray):
    """
    Saves SpectralNet cluster assignments to a file

    Args:
        assignments (np.ndarray): The assignments that obtained from SpectralNet
    """

    np.savetxt("cluster_assignments.csv", assignments.astype(int), fmt='%i', delimiter=',')


def create_weights_dir():
    """
    Creates a directory for the weights of the Autoencoder and the Siamese network
    """
    if not os.path.exists('weights'):
        os.makedirs('weights')


def get_affinity_matrix(X: torch.Tensor) -> torch.Tensor:
    """
    Computes the affinity matrix W

    Args:
        X (torch.Tensor):  Data

    Returns:
        torch.Tensor: Affinity matrix W
    """
    is_local = True
    n_neighbors = 30
    scale_k = 15
    Dx = torch.cdist(X,X)
    Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
    scale = compute_scale(Dis, k=scale_k, is_local=is_local)
    W = get_gaussian_kernel(Dx, scale, indices, device=torch.device("cpu"), is_local=is_local)
    return W

