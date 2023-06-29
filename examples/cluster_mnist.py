import sys
import os
import json
import torch
import random
import numpy as np

from data import load_data

from spectralnet import Metrics
from spectralnet import SpectralNet


class InvalidMatrixException(Exception):
    pass


def set_seed(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    x_train, x_test, y_train, y_test = load_data("mnist")

    if y_train is None:
        x_train = torch.cat([x_train, x_test])

    else:
        x_train = torch.cat([x_train, x_test])
        y_train = torch.cat([y_train, y_test])

    spectralnet = SpectralNet(
        n_clusters=10,
        should_use_ae=True,
        should_use_siamese=True,
    )
    spectralnet.fit(x_train)
    cluster_assignments = spectralnet.predict(x_train)
    embeddings = spectralnet.embeddings_

    if y_train is not None:
        y = y_train.detach().cpu().numpy()
        acc_score = Metrics.acc_score(cluster_assignments, y, n_clusters=10)
        nmi_score = Metrics.nmi_score(cluster_assignments, y)
        print(f"ACC: {np.round(acc_score, 3)}")
        print(f"NMI: {np.round(nmi_score, 3)}")

    return embeddings, cluster_assignments


if __name__ == "__main__":
    embeddings, assignments = main()
