import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange
from ._trainer import Trainer
from .._models import AEModel
from torch.utils.data import DataLoader, random_split


class AETrainer:
    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.ae_config = config
        self.lr = self.ae_config["lr"]
        self.epochs = self.ae_config["epochs"]
        self.min_lr = self.ae_config["min_lr"]
        self.lr_decay = self.ae_config["lr_decay"]
        self.patience = self.ae_config["patience"]
        self.architecture = self.ae_config["hiddens"]
        self.batch_size = self.ae_config["batch_size"]
        self.weights_dir = "spectralnet/_trainers/weights"
        self.weights_path = "spectralnet/_trainers/weights/ae_weights.pth"
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

    def train(self, X: torch.Tensor) -> AEModel:
        self.X = X.view(X.size(0), -1)
        self.criterion = nn.MSELoss()

        self.ae_net = AEModel(self.architecture, input_dim=self.X.shape[1]).to(
            self.device
        )

        self.optimizer = optim.Adam(self.ae_net.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        if os.path.exists(self.weights_path):
            self.ae_net.load_state_dict(torch.load(self.weights_path))
            return self.ae_net

        train_loader, valid_loader = self._get_data_loader()

        print("Training Autoencoder:")
        t = trange(self.epochs, leave=True)
        for epoch in t:
            train_loss = 0.0
            for batch_x in train_loader:
                batch_x = batch_x.to(self.device)
                batch_x = batch_x.view(batch_x.size(0), -1)
                self.optimizer.zero_grad()
                output = self.ae_net(batch_x)
                loss = self.criterion(output, batch_x)
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

        torch.save(self.ae_net.state_dict(), self.weights_path)
        return self.ae_net

    def validate(self, valid_loader: DataLoader) -> float:
        self.ae_net.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_x in valid_loader:
                batch_x = batch_x.to(self.device)
                batch_x = batch_x.view(batch_x.size(0), -1)
                output = self.ae_net(batch_x)
                loss = self.criterion(output, batch_x)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        return valid_loss

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        print("Embedding data ...")
        self.ae_net.eval()
        with torch.no_grad():
            X = X.view(X.size(0), -1)
            encoded_data = self.ae_net.encode(X.to(self.device))
        return encoded_data

    def _get_data_loader(self) -> tuple:
        trainset_len = int(len(self.X) * 0.9)
        validset_len = len(self.X) - trainset_len
        trainset, validset = random_split(self.X, [trainset_len, validset_len])
        train_loader = DataLoader(
            trainset, batch_size=self.ae_config["batch_size"], shuffle=True
        )
        valid_loader = DataLoader(
            validset, batch_size=self.ae_config["batch_size"], shuffle=False
        )
        return train_loader, valid_loader
