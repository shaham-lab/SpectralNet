import torch
import numpy as np

from data import load_data
from spectralnet import SpectralReduction


def main():
    x_train, x_test, y_train, y_test = load_data("twomoons")
    X = torch.cat([x_train, x_test])

    if y_train is not None:
        y = torch.cat([y_train, y_test])
    else:
        y = None

    spectralreduction = SpectralReduction(
        n_components=2,
        should_use_ae=False,
        should_use_siamese=False,
        spectral_batch_size=712,
        spectral_epochs=40,
        spectral_is_local_scale=False,
        spectral_n_nbg=8,
        spectral_scale_k=2,
        spectral_lr=1e-2,
        spectral_hiddens=[128, 128, 2],
    )

    X_new = spectralreduction.fit_transform(X)
    spectralreduction.visualize(X_new, y, n_components=1)


if __name__ == "__main__":
    main()
