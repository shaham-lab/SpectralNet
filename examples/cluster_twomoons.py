import torch
import numpy as np

from data import load_data

from spectralnet import Metrics
from spectralnet import SpectralNet


def main():
    x_train, x_test, y_train, y_test = load_data("twomoons")
    X = torch.cat([x_train, x_test])

    if y_train is not None:
        y = torch.cat([y_train, y_test])
    else:
        y = None

    spectralnet = SpectralNet(
        n_clusters=2,
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

    spectralnet.fit(X, y)
    cluster_assignments = spectralnet.predict(X)
    embeddings = spectralnet.embeddings_

    if y is not None:
        y = y.detach().cpu().numpy()
        acc_score = Metrics.acc_score(cluster_assignments, y, n_clusters=2)
        nmi_score = Metrics.nmi_score(cluster_assignments, y)
        print(f"ACC: {np.round(acc_score, 3)}")
        print(f"NMI: {np.round(nmi_score, 3)}")

    return embeddings, cluster_assignments


if __name__ == "__main__":
    embeddings, assignments = main()
