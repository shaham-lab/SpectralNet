import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors


def build_ann(X: torch.Tensor):
    """
    Builds approximate-nearest-neighbors object
    that can be used to calculate the k-nearest neighbors of a data-point

    Parameters
    ----------
    X : torch.Tensor
        Dataset.

    Returns
    -------
    None
    """

    X = X.view(X.size(0), -1)
    t = AnnoyIndex(X[0].shape[0], "euclidean")
    for i, x_i in enumerate(X):
        t.add_item(i, x_i)

    t.build(50)
    t.save("ann_index.ann")


def make_batch_for_sparse_grapsh(batch_x: torch.Tensor) -> torch.Tensor:
    """
    Computes a new batch of data points from the given batch (batch_x)
    in case that the graph-laplacian obtained from the given batch is sparse.
    The new batch is computed based on the nearest neighbors of 0.25
    of the given batch.

    Parameters
    ----------
    batch_x : torch.Tensor
        Batch of data points.

    Returns
    -------
    torch.Tensor
        New batch of data points.
    """

    batch_size = batch_x.shape[0]
    batch_size //= 5
    new_batch_x = batch_x[:batch_size]
    batch_x = new_batch_x
    n_neighbors = 5

    u = AnnoyIndex(batch_x[0].shape[0], "euclidean")
    u.load("ann_index.ann")
    for x in batch_x:
        x = x.detach().cpu().numpy()
        nn_indices = u.get_nns_by_vector(x, n_neighbors)
        nn_tensors = [u.get_item_vector(i) for i in nn_indices[1:]]
        nn_tensors = torch.tensor(nn_tensors)
        new_batch_x = torch.cat((new_batch_x, nn_tensors))

    return new_batch_x


def get_laplacian(W: torch.Tensor) -> np.ndarray:
    """
    Computes the unnormalized Laplacian matrix, given the affinity matrix W.

    Parameters
    ----------
    W : torch.Tensor
        Affinity matrix.

    Returns
    -------
    np.ndarray
        Laplacian matrix.
    """

    W = W.detach().cpu().numpy()
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L


def sort_laplacian(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sorts the columns and rows of the Laplacian by the true labels in order
    to see whether the sorted Laplacian is a block diagonal matrix.

    Parameters
    ----------
    L : np.ndarray
        Laplacian matrix.
    y : np.ndarray
        Labels.

    Returns
    -------
    np.ndarray
        Sorted Laplacian.
    """

    i = np.argsort(y)
    L = L[i, :]
    L = L[:, i]
    return L


def sort_matrix_rows(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sorts the rows of a matrix by a given order.

    Parameters
    ----------
    A : np.ndarray
        Numpy ndarray.
    y : np.ndarray
        True labels.

    Returns
    -------
    np.ndarray
        Sorted matrix.
    """

    i = np.argsort(y)
    A = A[i, :]
    return A


def get_eigenvalues(A: np.ndarray) -> np.ndarray:
    """
    Computes the eigenvalues of a given matrix A and sorts them in increasing order.

    Parameters
    ----------
    A : np.ndarray
        Numpy ndarray.

    Returns
    -------
    np.ndarray
        Sorted eigenvalues.
    """

    _, vals, _ = np.linalg.svd(A)
    sorted_vals = vals[np.argsort(vals)]
    return sorted_vals


def get_eigenvectors(A: np.ndarray) -> np.ndarray:
    """
    Computes the eigenvectors of a given matrix A and sorts them by the eigenvalues.

    Parameters
    ----------
    A : np.ndarray
        Numpy ndarray.

    Returns
    -------
    np.ndarray
        Sorted eigenvectors.
    """

    vecs, vals, _ = np.linalg.svd(A)
    vecs = vecs[:, np.argsort(vals)]
    return vecs


def plot_eigenvalues(vals: np.ndarray):
    """
    Plot the eigenvalues of the Laplacian.

    Parameters
    ----------
    vals : np.ndarray
        Eigenvalues.
    """

    rang = range(len(vals))
    plt.plot(rang, vals)
    plt.show()


def get_laplacian_eigenvectors(V: torch.Tensor, y: np.ndarray) -> np.ndarray:
    """
    Returns eigenvectors of the Laplacian when the data is sorted in increasing
    order by the true label.

    Parameters
    ----------
    V : torch.Tensor
        Eigenvectors matrix.
    y : np.ndarray
        True labels.

    Returns
    -------
    np.ndarray
        Sorted eigenvectors matrix and range.

    """

    V = sort_matrix_rows(V, y)
    rang = range(len(y))
    return V, rang


def plot_laplacian_eigenvectors(V: np.ndarray, y: np.ndarray):
    """
    Plot the eigenvectors of the Laplacian when the data is sorted in increasing
    order by the true label.

    Parameters
    ----------
    V : np.ndarray
        Eigenvectors matrix.
    y : np.ndarray
        True labels.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the plot.
    """

    V = sort_matrix_rows(V, y)
    rang = range(len(y))
    plt.plot(rang, V)
    plt.show()
    return plt


def plot_sorted_laplacian(W: torch.Tensor, y: np.ndarray):
    """
    Plot the block diagonal matrix obtained from the sorted Laplacian.

    Parameters
    ----------
    W : torch.Tensor
        Affinity matrix.
    y : np.ndarray
        True labels.
    """
    L = get_laplacian(W)
    L = sort_laplacian(L, y)
    plt.imshow(L, cmap="hot", norm=colors.LogNorm())
    plt.imshow(L, cmap="flag")
    plt.show()


def get_nearest_neighbors(
    X: torch.Tensor, Y: torch.Tensor = None, k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the distances and the indices of the k nearest neighbors of each data point.

    Parameters
    ----------
    X : torch.Tensor
        Batch of data points.
    Y : torch.Tensor, optional
        Defaults to None.
    k : int, optional
        Number of nearest neighbors to calculate. Defaults to 3.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Distances and indices of each data point.
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
    Computes the Grassmann distance between the subspaces spanned by the columns of A and B.

    Parameters
    ----------
    A : np.ndarray
        Numpy ndarray.
    B : np.ndarray
        Numpy ndarray.

    Returns
    -------
    float
        The Grassmann distance.
    """

    M = np.dot(np.transpose(A), B)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = 1 - np.square(s)
    grassmann = np.sum(s)
    return grassmann


def compute_scale(
    Dis: np.ndarray, k: int = 2, med: bool = True, is_local: bool = True
) -> np.ndarray:
    """
    Computes the scale for the Gaussian similarity function.

    Parameters
    ----------
    Dis : np.ndarray
        Distances of the k nearest neighbors of each data point.
    k : int, optional
        Number of nearest neighbors for the scale calculation. Relevant for global scale only.
    med : bool, optional
        Scale calculation method. Can be calculated by the median distance from a data point to its neighbors,
        or by the maximum distance. Defaults to True.
    is_local : bool, optional
        Local distance (different for each data point), or global distance. Defaults to True.

    Returns
    -------
    np.ndarray
        Scale (global or local).
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


def get_gaussian_kernel(
    D: torch.Tensor, scale, Ids: np.ndarray, device: torch.device, is_local: bool = True
) -> torch.Tensor:
    """
    Computes the Gaussian similarity function according to a given distance matrix D and a given scale.

    Parameters
    ----------
    D : torch.Tensor
        Distance matrix.
    scale :
        Scale.
    Ids : np.ndarray
        Indices of the k nearest neighbors of each sample.
    device : torch.device
        Defaults to torch.device("cpu").
    is_local : bool, optional
        Determines whether the given scale is global or local. Defaults to True.

    Returns
    -------
    torch.Tensor
        Matrix W with Gaussian similarities.
    """

    if not is_local:
        # global scale
        W = torch.exp(-torch.pow(D, 2) / (scale**2))
    else:
        # local scales
        W = torch.exp(
            -torch.pow(D, 2).to(device)
            / (torch.tensor(scale).float().to(device).clamp_min(1e-7) ** 2)
        )
    if Ids is not None:
        n, k = Ids.shape
        mask = torch.zeros([n, n]).to(device=device)
        for i in range(len(Ids)):
            mask[i, Ids[i]] = 1
        W = W * mask
    sym_W = (W + torch.t(W)) / 2.0
    return sym_W


def get_t_kernel(
    D: torch.Tensor, Ids: np.ndarray, device: torch.device, is_local: bool = True
) -> torch.Tensor:
    """
    Computes the t similarity function according to a given distance matrix D and a given scale.

    Parameters
    ----------
    D : torch.Tensor
        Distance matrix.
    Ids : np.ndarray
        Indices of the k nearest neighbors of each sample.
    device : torch.device
        Defaults to torch.device("cpu").
    is_local : bool, optional
        Determines whether the given scale is global or local. Defaults to True.

    Returns
    -------
    torch.Tensor
        Matrix W with t similarities.
    """

    W = torch.pow(1 + torch.pow(D, 2), -1)
    if Ids is not None:
        n, k = Ids.shape
        mask = torch.zeros([n, n]).to(device=device)
        for i in range(len(Ids)):
            mask[i, Ids[i]] = 1
        W = W * mask
    sym_W = (W + W.T) / 2.0
    return sym_W


def plot_data_by_assignments(X, assignments: np.ndarray):
    """
    Plots the data with the assignments obtained from SpectralNet. Relevant only for 2D data.

    Parameters
    ----------
    X :
        Data.
    assignments : np.ndarray
        Cluster assignments.
    """

    plt.scatter(X[:, 0], X[:, 1], c=assignments)
    plt.show()


def calculate_cost_matrix(C: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Calculates the cost matrix for the Munkres algorithm.

    Parameters
    ----------
    C : np.ndarray
        Confusion matrix.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Cost matrix.
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
    Gets the cluster labels from their indices.

    Parameters
    ----------
    indices : np.ndarray
        Indices of the clusters.

    Returns
    -------
    np.ndarray
        Cluster labels.
    """

    num_clusters = len(indices)
    cluster_labels = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def write_assignments_to_file(assignments: np.ndarray):
    """
    Saves SpectralNet cluster assignments to a file.

    Parameters
    ----------
    assignments : np.ndarray
        The assignments that obtained from SpectralNet.
    """

    np.savetxt(
        "cluster_assignments.csv", assignments.astype(int), fmt="%i", delimiter=","
    )


def create_weights_dir():
    """
    Creates a directory for the weights of the Autoencoder and the Siamese network
    """
    if not os.path.exists("weights"):
        os.makedirs("weights")
