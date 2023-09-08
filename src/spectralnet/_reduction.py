import torch
import numpy as np
import matplotlib.pyplot as plt

from ._utils import *
from ._cluster import SpectralNet
from sklearn.cluster import KMeans
from ._metrics import Metrics


class SpectralReduction:
    def __init__(
        self,
        n_components: int,
        should_use_ae: bool = False,
        should_use_siamese: bool = False,
        is_sparse_graph: bool = False,
        ae_hiddens: list = [512, 512, 2048, 10],
        ae_epochs: int = 40,
        ae_lr: float = 1e-3,
        ae_lr_decay: float = 0.1,
        ae_min_lr: float = 1e-7,
        ae_patience: int = 10,
        ae_batch_size: int = 256,
        siamese_hiddens: list = [1024, 1024, 512, 10],
        siamese_epochs: int = 30,
        siamese_lr: float = 1e-3,
        siamese_lr_decay: float = 0.1,
        siamese_min_lr: float = 1e-7,
        siamese_patience: int = 10,
        siamese_n_nbg: int = 2,
        siamese_use_approx: bool = False,
        siamese_batch_size: int = 128,
        spectral_hiddens: list = [1024, 1024, 512, 10],
        spectral_epochs: int = 30,
        spectral_lr: float = 1e-3,
        spectral_lr_decay: float = 0.1,
        spectral_min_lr: float = 1e-8,
        spectral_patience: int = 10,
        spectral_batch_size: int = 1024,
        spectral_n_nbg: int = 30,
        spectral_scale_k: int = 15,
        spectral_is_local_scale: bool = True,
    ):
        """SpectralNet is a class for implementing a Deep learning model that performs spectral clustering.
        This model optionally utilizes Autoencoders (AE) and Siamese networks for training.

        Parameters
        ----------
        n_components : int
            The number of components to keep.

        should_use_ae : bool, optional (default=False)
            Specifies whether to use the Autoencoder (AE) network as part of the training process.

        should_use_siamese : bool, optional (default=False)
                Specifies whether to use the Siamese network as part of the training process.

        is_sparse_graph : bool, optional (default=False)
            Specifies whether the graph Laplacian created from the data is sparse.

        ae_hiddens : list, optional (default=[512, 512, 2048, 10])
            The number of hidden units in each layer of the Autoencoder network.

        ae_epochs : int, optional (default=30)
            The number of epochs to train the Autoencoder network.

        ae_lr : float, optional (default=1e-3)
            The learning rate for the Autoencoder network.

        ae_lr_decay : float, optional (default=0.1)
            The learning rate decay factor for the Autoencoder network.

        ae_min_lr : float, optional (default=1e-7)
            The minimum learning rate for the Autoencoder network.

        ae_patience : int, optional (default=10)
            The number of epochs to wait before reducing the learning rate for the Autoencoder network.

        ae_batch_size : int, optional (default=256)
            The batch size used during training of the Autoencoder network.

        siamese_hiddens : list, optional (default=[1024, 1024, 512, 10])
            The number of hidden units in each layer of the Siamese network.

        siamese_epochs : int, optional (default=30)
            The number of epochs to train the Siamese network.

        siamese_lr : float, optional (default=1e-3)
            The learning rate for the Siamese network.

        siamese_lr_decay : float, optional (default=0.1)
            The learning rate decay factor for the Siamese network.

        siamese_min_lr : float, optional (default=1e-7)
            The minimum learning rate for the Siamese network.

        siamese_patience : int, optional (default=10)
            The number of epochs to wait before reducing the learning rate for the Siamese network.

        siamese_n_nbg : int, optional (default=2)
            The number of nearest neighbors to consider as 'positive' pairs by the Siamese network.

        siamese_use_approx : bool, optional (default=False)
            Specifies whether to use Annoy instead of KNN for computing nearest neighbors,
            particularly useful for large datasets.

        siamese_batch_size : int, optional (default=256)
            The batch size used during training of the Siamese network.

        spectral_hiddens : list, optional (default=[1024, 1024, 512, 10])
            The number of hidden units in each layer of the Spectral network.

        spectral_epochs : int, optional (default=30)
            The number of epochs to train the Spectral network.

        spectral_lr : float, optional (default=1e-3)
            The learning rate for the Spectral network.

        spectral_lr_decay : float, optional (default=0.1)
            The learning rate decay factor"""

        self.n_components = n_components
        self.should_use_ae = should_use_ae
        self.should_use_siamese = should_use_siamese
        self.is_sparse_graph = is_sparse_graph
        self.ae_hiddens = ae_hiddens
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.ae_lr_decay = ae_lr_decay
        self.ae_min_lr = ae_min_lr
        self.ae_patience = ae_patience
        self.ae_batch_size = ae_batch_size
        self.siamese_hiddens = siamese_hiddens
        self.siamese_epochs = siamese_epochs
        self.siamese_lr = siamese_lr
        self.siamese_lr_decay = siamese_lr_decay
        self.siamese_min_lr = siamese_min_lr
        self.siamese_patience = siamese_patience
        self.siamese_n_nbg = siamese_n_nbg
        self.siamese_use_approx = siamese_use_approx
        self.siamese_batch_size = siamese_batch_size
        self.spectral_hiddens = spectral_hiddens
        self.spectral_epochs = spectral_epochs
        self.spectral_lr = spectral_lr
        self.spectral_lr_decay = spectral_lr_decay
        self.spectral_min_lr = spectral_min_lr
        self.spectral_patience = spectral_patience
        self.spectral_n_nbg = spectral_n_nbg
        self.spectral_scale_k = spectral_scale_k
        self.spectral_is_local_scale = spectral_is_local_scale
        self.spectral_batch_size = spectral_batch_size
        self.X_new = None

    def _fit(self, X: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """Fit the SpectralNet model to the input data.

        Parameters
        ----------
        X : torch.Tensor
            The input data of shape (n_samples, n_features).

        y: torch.Tensor
            The labels of the input data of shape (n_samples,).

        Returns
        -------
        np.ndarray
            The fitted embeddings of shape (n_samples, n_components).
        """
        self._spectralnet = SpectralNet(
            n_clusters=self.n_components,
            should_use_ae=self.should_use_ae,
            should_use_siamese=self.should_use_siamese,
            is_sparse_graph=self.is_sparse_graph,
            ae_hiddens=self.ae_hiddens,
            ae_epochs=self.ae_epochs,
            ae_lr=self.ae_lr,
            ae_lr_decay=self.ae_lr_decay,
            ae_min_lr=self.ae_min_lr,
            ae_patience=self.ae_patience,
            ae_batch_size=self.ae_batch_size,
            siamese_hiddens=self.siamese_hiddens,
            siamese_epochs=self.siamese_epochs,
            siamese_lr=self.siamese_lr,
            siamese_lr_decay=self.siamese_lr_decay,
            siamese_min_lr=self.siamese_min_lr,
            siamese_patience=self.siamese_patience,
            siamese_n_nbg=self.siamese_n_nbg,
            siamese_use_approx=self.siamese_use_approx,
            siamese_batch_size=self.siamese_batch_size,
            spectral_hiddens=self.spectral_hiddens,
            spectral_epochs=self.spectral_epochs,
            spectral_lr=self.spectral_lr,
            spectral_lr_decay=self.spectral_lr_decay,
            spectral_min_lr=self.spectral_min_lr,
            spectral_patience=self.spectral_patience,
            spectral_n_nbg=self.spectral_n_nbg,
            spectral_scale_k=self.spectral_scale_k,
            spectral_is_local_scale=self.spectral_is_local_scale,
            spectral_batch_size=self.spectral_batch_size,
        )

        self._spectralnet.fit(X, y)

    def _predict(self, X: torch.Tensor) -> np.ndarray:
        """Predict embeddings for the input data using the fitted SpectralNet model.

        Parameters
        ----------
        X : torch.Tensor
            The input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The predicted embeddings of shape (n_samples, n_components).
        """
        self._spectralnet.predict(X)
        return self._spectralnet.embeddings_

    def _transform(self, X: torch.Tensor) -> np.ndarray:
        """Transform the input data into embeddings using the fitted SpectralNet model.

        Parameters
        ----------
        X : torch.Tensor
            The input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The transformed embeddings of shape (n_samples, n_components).
        """
        return self._predict(X)

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None) -> np.ndarray:
        """Fit the SpectralNet model to the input data and transform it into embeddings.

        This is a convenience method that combines the fit and transform steps.

        Parameters
        ----------
        X : torch.Tensor
            The input data of shape (n_samples, n_features).

        y: torch.Tensor
            The labels of the input data of shape (n_samples,).

        Returns
        -------
        np.ndarray
            The fitted and transformed embeddings of shape (n_samples, n_components).
        """
        self._fit(X, y)
        return self._transform(X)

    def _get_laplacian_of_small_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Get the Laplacian of a small batch of the input data

        Parameters
        ----------

        batch : torch.Tensor
            A small batch of the input data of shape (batch_size, n_features).

        Returns
        -------
        np.ndarray
            The Laplacian of the small batch of the input data.



        """

        W = get_affinity_matrix(batch, self.spectral_n_nbg, self._spectralnet.device)
        L = get_laplacian(W)
        return L

    def _remove_smallest_eigenvector(self, V: np.ndarray) -> np.ndarray:
        """Remove the constant eigenvector from the eigenvectors of the Laplacian of a small batch of the input data.


        Parameters
        ----------
        V : np.ndarray
            The eigenvectors of the Laplacian of a small batch of the input data.


        Returns
        -------
        np.ndarray
            The eigenvectors of the Laplacian of a small batch of the input data without the constant eigenvector.
        """

        batch_raw, batch_encoded = self._spectralnet.get_random_batch()
        L_batch = self._get_laplacian_of_small_batch(batch_encoded)
        V_batch = self._predict(batch_raw)
        eigenvalues = np.diag(V_batch.T @ L_batch @ V_batch)
        indices = np.argsort(eigenvalues)
        smallest_index = indices[0]
        V = V[:, np.arange(V.shape[1]) != smallest_index]
        V = V[
            :,
            (np.arange(V.shape[1]) == indices[1])
            | (np.arange(V.shape[1]) == indices[2]),
        ]

        return V

    def visualize(
        self, V: np.ndarray, y: torch.Tensor = None, n_components: int = 1
    ) -> None:
        """Visualize the embeddings of the input data using the fitted SpectralNet model.

        Parameters
        ----------
        V : torch.Tensor
            The reduced data of shape (n_samples, n_features) to be visualized.
        y : torch.Tensor
            The input labels of shape (n_samples,).
        """
        V = self._remove_smallest_eigenvector(V)
        print(V.shape)

        plot_laplacian_eigenvectors(V, y)
        cluster_labels = self._get_clusters_by_kmeans(V)
        acc = Metrics.acc_score(cluster_labels, y.detach().cpu().numpy(), n_clusters=10)
        print("acc with 2 components: ", acc)

        if n_components > 1:
            x_axis = V[:, 0]
            y_axis = V[:, 1]

        elif n_components == 1:
            x_axis = V
            y_axis = np.zeros_like(V)

        else:
            raise ValueError(
                "n_components must be a positive integer (greater than 0))"
            )

        if y is None:
            plt.scatter(x_axis, y_axis)
        else:
            plt.scatter(x_axis, y_axis, c=y, cmap="tab10", s=3)

        plt.show()

    def _get_clusters_by_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        """Performs k-means clustering on the spectral-embedding space.

        Parameters
        ----------
        embeddings : np.ndarray
            The spectral-embedding space.

        Returns
        -------
        np.ndarray
            The cluster assignments for the given data.
        """

        kmeans = KMeans(n_clusters=self.n_components, n_init=10).fit(embeddings)
        cluster_assignments = kmeans.predict(embeddings)
        return cluster_assignments
