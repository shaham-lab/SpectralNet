import torch
import numpy as np

from data import load_data

from spectralnet import Metrics
from spectralnet import SpectralReduction


def main():
    x_train, x_test, y_train, y_test = load_data("mnist")
    X = torch.cat([x_train, x_test])

    if y_train is not None:
        y = torch.cat([y_train, y_test])
    else:
        y = None

    spectralreduction = SpectralReduction(
        n_components=3,
        should_use_ae=True,
        should_use_siamese=True,
        spectral_hiddens=[512, 512, 2048, 3],
    )

    X_new = spectralreduction.fit_transform(X)
    spectralreduction.visualize(X_new, y, n_components=2)


if __name__ == "__main__":
    main()
