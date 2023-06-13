import torch
import numpy as np

from AETrainer import *
from SiameseTrainer import *
from SpectralTrainer import *
from sklearn.cluster import KMeans


class SpectralNet:
    def __init__(self, n_clusters: int, config: dict):
        """
        Args:
            n_clusters (int):   The dimension of the projection subspace
            config (dict):      The configuration dictionary
        """

        self.n_clusters = n_clusters
        self.config = config
        self.embeddings_ = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def fit(self, X: torch.Tensor ,y: torch.Tensor = None):
        """
        Performs the main training loop for the SpectralNet model.

        Args:
            X (torch.Tensor):   Data to train the networks on
            y (torch.Tensor):   Labels in case there are any. Defaults to None.
        """

        should_use_ae = self.config["should_use_ae"]
        should_use_siamese = self.config["should_use_siamese"]
        create_weights_dir()

        if should_use_ae:
            ae_trainer = AETrainer(self.config, self.device)
            self.ae_net = ae_trainer.train(X)
            X = ae_trainer.embed(X)
        
        if should_use_siamese:
            siamese_trainer = SiameseTrainer(self.config, self.device)
            self.siamese_net = siamese_trainer.train(X)
        else:
            self.siamese_net = None

        is_sparse = self.config["is_sparse_graph"]
        if is_sparse:
            build_ann(X)
        spectral_trainer = SpectralTrainer(self.config, self.device, is_sparse=is_sparse)
        self.spec_net = spectral_trainer.train(X, y, self.siamese_net)
        
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Predicts the cluster assignments for the given data.
        
        Args:
            X (torch.Tensor):   Data to be clustered
        
        Returns:
            np.ndarray:  The cluster assignments for the given data

        """      
        X = X.view(X.size(0), -1)
        X = X.to(self.device)
        should_use_ae = self.config["should_use_ae"]
        
        with torch.no_grad():
            if should_use_ae:
                X = self.ae_net.encoder(X)
            self.embeddings_ = self.spec_net(X, should_update_orth_weights=False).detach().cpu().numpy()
        
        cluster_assignments = self._get_clusters_by_kmeans(self.embeddings_)
        return cluster_assignments

    
    def _get_clusters_by_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Performs k-means clustering on the spectral-embedding space.

        Args:
            embeddings (np.ndarray):   the spectral-embedding space

        Returns:
            np.ndarray:  the cluster assignments for the given data
        """
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10).fit(embeddings)
        cluster_assignments = kmeans.predict(embeddings)
        return cluster_assignments
    
