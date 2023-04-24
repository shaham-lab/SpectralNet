import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import *
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, random_split, TensorDataset



class SpectralNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(SpectralNetModel, self).__init__()
        self.architecture = architecture
        self.num_of_layers = self.architecture["n_layers"]
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        
        current_dim = self.input_dim
        for layer, dim in self.architecture.items():
            next_dim = dim
            if layer == "n_layers":
                continue
            if layer == "output_dim":
                layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.Tanh())
                self.layers.append(layer)
            else:
                layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
                self.layers.append(layer)
                current_dim = next_dim
  

    def forward(self, x: torch.Tensor, is_orthonorm: bool = True) -> torch.Tensor:
        """
        This function performs the forward pass of the model.
        If is_orthonorm is True, the output of the network is orthonormalized
        using the Cholesky decomposition.

        Args:
            x (torch.Tensor):               The input tensor
            is_orthonorm (bool, optional):  Whether to orthonormalize the output or not. 
                                            Defaults to True.

        Returns:
            torch.Tensor: The output tensor
        """

        for layer in self.layers:
            x = layer(x)

        Y_tilde = x
        if is_orthonorm:
            m = Y_tilde.shape[0]
            to_factorize = torch.mm(Y_tilde.t(), Y_tilde)
            
            try:
                L = torch.linalg.cholesky(to_factorize, upper=False)
            except torch._C._LinAlgError:
                to_factorize += 0.1 * torch.eye(to_factorize.shape[0])
                L = torch.linalg.cholesky(to_factorize, upper=False)

            L_inverse = torch.inverse(L)
            self.orthonorm_weights = np.sqrt(m) * L_inverse.t()

        Y = torch.mm(Y_tilde, self.orthonorm_weights)
        return Y



class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()
    
    def forward(self, W: torch.Tensor, Y: torch.Tensor , is_normalized: bool = False) -> torch.Tensor:
        """
        This function computes the loss of the SpectralNet model.
        The loss is the rayleigh quotient of the Laplacian matrix obtained from W, 
        and the orthonormalized output of the network.

        Args:
            W (torch.Tensor):               Affinity matrix
            Y (torch.Tensor):               Output of the network
            is_normalized (bool, optional): Whether to use the normalized Laplacian matrix or not.

        Returns:
            torch.Tensor: The loss
        """
        m = Y.size(0)
        if is_normalized:
            D = torch.sum(W, dim=1)
            Y = Y / D[:, None]

        Dy = torch.cdist(Y, Y)
        loss = torch.sum(W * Dy.pow(2)) / (2 * m)

        return loss


class SpectralTrainer:
    def __init__(self, config: dict, device: torch.device, is_sparse: bool = False):
        """
        This class is responsible for training the SpectralNet model.

        Args:
            config (dict):                  The configuration dictionary
            device (torch.device):          The device to use for training
            is_sparse (bool, optional):     Whether the graph-laplacian obtained from a mini-batch is sparse or not.
                                            In case it is sparse, we build the batch by taking 1/5 of the original random batch,
                                            and then we add to each sample 4 of its nearest neighbors. Defaults to False.
        """

        self.device = device
        self.is_sparse = is_sparse
        self.spectral_config = config["spectral"]

        self.lr = self.spectral_config["lr"]
        self.epochs = self.spectral_config["epochs"]
        self.lr_decay = self.spectral_config["lr_decay"]
        self.patience = self.spectral_config["patience"]
        self.batch_size = self.spectral_config["batch_size"]
        self.architecture = self.spectral_config["architecture"]
    
    def train(self, X: torch.Tensor, y: torch.Tensor, siamese_net: nn.Module = None) -> SpectralNetModel:
        """
        This function trains the SpectralNet model.

        Args:
            X (torch.Tensor):                       The dataset to train on
            y (torch.Tensor):                       The labels of the dataset in case there are any
            siamese_net (nn.Module, optional):      The siamese network to use for computing the affinity matrix.

        Returns:
            SpectralNetModel: The trained SpectralNet model
        """

        self.X = X.view(X.size(0), -1)
        self.y = y
        self.counter = 0
        self.siamese_net = siamese_net
        self.criterion = SpectralNetLoss()
        self.spectral_net = SpectralNetModel(self.architecture, input_dim=self.X.shape[1]).to(self.device)
        self.optimizer = optim.Adam(self.spectral_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode='min', 
                                                              factor=self.lr_decay, 
                                                              patience=self.patience)

        
        train_loader, ortho_loader, valid_loader = self._get_data_loader()
        
        print("Training SpectralNet:")
        for epoch in range(self.epochs):
            train_loss = 0.0
            for (X_grad, y_grad), (X_orth, _) in zip(train_loader, ortho_loader):
                X_grad = X_grad.to(device=self.device)
                X_grad = X_grad.view(X_grad.size(0), -1)
                X_orth = X_orth.to(device=self.device)
                X_orth = X_orth.view(X_orth.size(0), -1)

                if self.is_sparse:
                    X_grad = make_batch_for_sparse_grapsh(X_grad)
                    X_orth = make_batch_for_sparse_grapsh(X_orth)

                # Orthogonalization step
                self.spectral_net.eval()
                self.spectral_net(X_orth, is_orthonorm=True)
                
                # Gradient step
                self.spectral_net.train()
                self.optimizer.zero_grad()
                
                if self.is_sparse:
                    X_grad = make_batch_for_sparse_grapsh(X_grad)

                Y = self.spectral_net(X_grad, is_orthonorm=False)
                if self.siamese_net is not None:
                    with torch.no_grad():
                        X_grad = self.siamese_net.forward_once(X_grad)

                W = self._get_affinity_matrix(X_grad)

                loss = self.criterion(W, Y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            
            # Validation step
            valid_loss = self.validate(valid_loader)
            self.scheduler.step(valid_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.spectral_config["min_lr"]: break
            print("Epoch: {}/{}, Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".
            format(epoch + 1, self.epochs, train_loss, valid_loss, current_lr))
        
        return self.spectral_net
    
    def validate(self, valid_loader: DataLoader) -> float:
        """
        This function validates the SpectralNet model during the training process.

        Args:
            valid_loader (DataLoader):  The validation data loader

        Returns:
            float: The validation loss
        """

        valid_loss = 0.0
        self.spectral_net.eval()
        with torch.no_grad():
            for batch in valid_loader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)

                if self.is_sparse:
                    X = make_batch_for_sparse_grapsh(X)
                    
                Y = self.spectral_net(X, is_orthonorm=False)
                with torch.no_grad():
                    if self.siamese_net is not None:
                        X = self.siamese_net.forward_once(X)

                W = self._get_affinity_matrix(X)

                loss = self.criterion(W, Y)
                valid_loss += loss.item()
        
        self.counter += 1

        valid_loss /= len(valid_loader)
        return valid_loss
            
    
    def _get_affinity_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        This function computes the affinity matrix W using the Gaussian kernel.

        Args:
            X (torch.Tensor):   The input data

        Returns:
            torch.Tensor: The affinity matrix W
        """

        is_local = self.spectral_config["is_local_scale"]
        n_neighbors = self.spectral_config["n_neighbors"]
        scale_k = self.spectral_config["scale_k"]
        Dx = torch.cdist(X,X)
        Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
        scale = compute_scale(Dis, k=scale_k, is_local=is_local)
        W = get_gaussian_kernel(Dx, scale, indices, device=self.device, is_local=is_local)
        return W


    def _get_data_loader(self) -> tuple:
        """
        This function returns the data loaders for training, validation and testing.

        Returns:
            tuple:  The data loaders
        """
        if self.y is None:
            self.y = torch.zeros(len(self.X))
        train_size = int(0.9 * len(self.X))
        valid_size = len(self.X) - train_size
        dataset = TensorDataset(self.X, self.y)
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        ortho_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, ortho_loader, valid_loader



class ReduceLROnAvgLossPlateau(_LRScheduler):
    def __init__(self, optimizer, factor=0.1, patience=10, min_lr=0, verbose=False, min_delta=1e-4):
        """
        Custom ReduceLROnPlateau scheduler that uses the average loss instead of the loss of the last epoch.

        Args:
            optimizer (_type_):             The optimizer
            factor (float, optional):       factor by which the learning rate will be reduced. 
                                            new_lr = lr * factor. Defaults to 0.1.
            patience (int, optional):       number of epochs with no average improvement after 
                                            which learning rate will be reduced.
            min_lr (int, optional):         A lower bound on the learning rate of all param groups.
            verbose (bool, optional):       If True, prints a message to stdout for each update.
            min_delta (_type_, optional):   threshold for measuring the new optimum, to only focus on
                                            significant changes. Defaults to 1e-4.
        """

        self.factor = factor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.best = 1e5
        self.avg_losses = []
        self.min_lr = min_lr
        super(ReduceLROnAvgLossPlateau, self).__init__(optimizer)

    def get_lr(self):
        return [base_lr * self.factor ** self.num_bad_epochs
                for base_lr in self.base_lrs]

    def step(self, loss=1.0,  epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        

        current_loss = loss
        if len(self.avg_losses) < self.patience:
            self.avg_losses.append(current_loss)
        else:
            self.avg_losses.pop(0)
            self.avg_losses.append(current_loss)
        avg_loss = sum(self.avg_losses) / len(self.avg_losses)
        if avg_loss < self.best - self.min_delta:
            self.best = avg_loss
            self.wait = 0
        else:
            if self.wait >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = float(param_group['lr'])
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        param_group['lr'] = new_lr
                        if self.verbose:
                            print(f'Epoch {epoch}: reducing learning rate to {new_lr}.')
                self.wait = 0
            self.wait += 1