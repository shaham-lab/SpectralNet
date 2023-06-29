import os
import torch
import numpy as np
import torch.optim as optim

from tqdm import trange
from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, random_split

from ._trainer import Trainer
from .._models import SiameseNetModel
from .._losses import ContrastiveLoss


class SiameseDataset:
    def __init__(self, pairs: list):
        """
        Initializes a Siamese dataset.

        Parameters
        ----------
        pairs : list
            A list of tuples containing the pairs of data
            and their labels.
        """
        self.pairs = pairs

    def __getitem__(self, index: int):
        x1 = self.pairs[index][0]
        x2 = self.pairs[index][1]
        label = self.pairs[index][2]
        return x1, x2, label

    def __len__(self):
        return len(self.pairs)


class SiameseTrainer:
    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.siamese_config = config
        self.lr = self.siamese_config["lr"]
        self.n_nbg = self.siamese_config["n_nbg"]
        self.min_lr = self.siamese_config["min_lr"]
        self.epochs = self.siamese_config["epochs"]
        self.lr_decay = self.siamese_config["lr_decay"]
        self.patience = self.siamese_config["patience"]
        self.architecture = self.siamese_config["hiddens"]
        self.batch_size = self.siamese_config["batch_size"]
        self.use_approx = self.siamese_config["use_approx"]
        self.weights_path = "./weights/siamese_weights.pth"

    def train(self, X: torch.Tensor) -> SiameseNetModel:
        self.X = X.view(X.size(0), -1)
        # self.X = X

        self.criterion = ContrastiveLoss()
        self.siamese_net = SiameseNetModel(
            self.architecture, input_dim=self.X.shape[1]
        ).to(self.device)

        self.optimizer = optim.Adam(self.siamese_net.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        if os.path.exists(self.weights_path):
            self.siamese_net.load_state_dict(torch.load(self.weights_path))
            return self.siamese_net

        train_loader, valid_loader = self._get_data_loader()

        print("Training Siamese Network:")
        t = trange(self.epochs, leave=True)
        self.siamese_net.train()
        for epoch in t:
            train_loss = 0.0
            for x1, x2, label in train_loader:
                x1 = x1.to(self.device)
                x1 = x1.view(x1.size(0), -1)
                x2 = x2.to(self.device)
                x2 = x2.view(x2.size(0), -1)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                output1, output2 = self.siamese_net(x1, x2)
                loss = self.criterion(output1, output2, label)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            valid_loss = self.validate(valid_loader)
            self.scheduler.step(valid_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            if current_lr <= self.min_lr:
                break
            t.set_description(
                "Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(
                    train_loss, valid_loss, current_lr
                )
            )
            t.refresh()

        return self.siamese_net

    def validate(self, valid_loader: DataLoader) -> float:
        valid_loss = 0.0
        self.siamese_net.eval()
        with torch.no_grad():
            for x1, x2, label in valid_loader:
                x1 = x1.to(self.device)
                x1 = x1.view(x1.size(0), -1)
                x2 = x2.to(self.device)
                x2 = x2.view(x2.size(0), -1)
                label = label.to(self.device)
                output1, output2 = self.siamese_net(x1, x2)
                loss = self.criterion(output1, output2, label)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        return valid_loss

    def _get_knn_pairs(self) -> list:
        """Gets the pairs of data points to be used for training the siamese network.

        Parameters
        ----------
        None

        Returns
        -------
        list
            A list of pairs of data points.

        Notes
        -----
        The pairs are chosen such that each data point has n_neighbors positive pairs
        and n_neighbors negative pairs where the neighbors are chosen using KNN.
        """

        pairs = []
        X = self.X.detach().cpu().numpy()
        data_indices = np.arange(len(X))
        n_neighbors = self.n_nbg
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree").fit(
            X
        )
        _, neighbors_indices = nbrs.kneighbors(X)

        for i in range(len(X)):
            non_neighbors_indices = np.delete(data_indices, neighbors_indices[i])
            non_neighbors_random_chosen_indices = np.random.choice(
                non_neighbors_indices, n_neighbors
            )

            positive_pairs = [
                [self.X[i], self.X[n], 1]
                for n in neighbors_indices[i][1 : n_neighbors + 1]
            ]
            negative_pairs = [
                [self.X[i], self.X[n], 0] for n in non_neighbors_random_chosen_indices
            ]

            pairs += positive_pairs
            pairs += negative_pairs

        return pairs

    def _get_approx_nn_pairs(self) -> list:
        """Gets the pairs of data points to be used for training the siamese network.

        Parameters
        ----------
        None

        Returns
        -------
        list
            A list of pairs of data points.

        Notes
        -----
        The pairs are chosen such that each data point has 1 neighbor from its nearest n_neighbors
        neighbors and 1 neighbor from the rest of the data points. The neighbors are chosen using
        approximate nearest neighbors using the Annoy library.
        """

        pairs = []
        n_samples = self.X.shape[0]
        n_neighbors = self.n_nbg
        indices = torch.randperm(self.X.shape[0])[:n_samples]
        x_train = self.X[indices]
        X_numpy = self.X[indices].detach().cpu().numpy()
        data_indices = np.arange(len(x_train))

        ann = AnnoyIndex(X_numpy.shape[1], "euclidean")
        for i, x_ in enumerate(X_numpy):
            ann.add_item(i, x_)
        ann.build(50)

        neighbors_indices = np.empty((len(X_numpy), n_neighbors + 1))
        for i in range(len(X_numpy)):
            nn_i = ann.get_nns_by_item(i, n_neighbors + 1, include_distances=False)
            neighbors_indices[i, :] = np.array(nn_i)
        neighbors_indices = neighbors_indices.astype(int)

        print("Building dataset for the siamese network ...")
        for i in range(len(X_numpy)):
            non_neighbors_indices = np.delete(data_indices, neighbors_indices[i])

            neighbor_idx = np.random.choice(neighbors_indices[i][1:], 1)
            non_nbr_idx = np.random.choice(non_neighbors_indices, 1)

            positive_pairs = [[x_train[i], x_train[neighbor_idx], 1]]
            negative_pairs = [[x_train[i], x_train[non_nbr_idx], 0]]

            pairs += positive_pairs
            pairs += negative_pairs

        return pairs

    def _get_pairs(self) -> list:
        """Gets the pairs of data points to be used for training the siamese network.

        Parameters
        ----------
        None

        Returns
        -------
        list
            A list of pairs of data points.

        Notes
        -----
        This method internally calls either _get_knn_pairs() or _get_approx_nn_pairs() based on the value
        of the 'use_approx' attribute.
        """

        should_use_approx = self.use_approx
        if should_use_approx:
            return self._get_approx_nn_pairs()
        else:
            return self._get_knn_pairs()

    def _get_data_loader(self) -> tuple:
        """
        Splits the data into train and validation sets and returns the corresponding data loaders.

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            A tuple containing the train and validation data loaders.

        Notes
        -----
        This function splits the data into train and validation sets and creates data loaders for them.
        The train and validation sets are obtained by randomly splitting the siamese dataset.
        The train and validation data loaders are created using DataLoader from the PyTorch library.
        """

        pairs = self._get_pairs()
        siamese_dataset = SiameseDataset(pairs)
        siamese_trainset_len = int(len(siamese_dataset) * 0.9)
        siamese_validset_len = len(siamese_dataset) - siamese_trainset_len
        siamese_trainset, siamese_validset = random_split(
            siamese_dataset, [siamese_trainset_len, siamese_validset_len]
        )
        siamese_trainloader = DataLoader(
            siamese_trainset, batch_size=self.siamese_config["batch_size"], shuffle=True
        )
        siamese_validloader = DataLoader(
            siamese_validset,
            batch_size=self.siamese_config["batch_size"],
            shuffle=False,
        )
        return siamese_trainloader, siamese_validloader
