# SpectralNet

<p align="center">
    <img src="https://github.com/shaham-lab/SpectralNet/blob/main/figures/twomoons.png">

SpectralNet is a Python package that performs spectral clustering with deep neural networks.<br><br>
This package is based on the following paper - [SpectralNet](https://openreview.net/pdf?id=HJ_aoCyRZ)

## Installation

You can install the latest package version via

```bash
pip install spectralnet
```

## Usage

The basic functionality is quite intuitive and easy to use, e.g.,

```python
from spectralnet import SpectralNet

spectralnet = SpectralNet(n_clusters=10)
spectralnet.fit(X) # X is the dataset and it should be a torch.Tensor
cluster_assignments = spectralnet.predict(X) # Get the final assignments to clusters
```

If you have labels to your dataset and you want to measure ACC and NMI you can do the following:

```python
from spectralnet import SpectralNet
from spectralnet import Metrics


spectralnet = SpectralNet(n_clusters=2)
spectralnet.fit(X, y) # X is the dataset and it should be a torch.Tensor
cluster_assignments = spectralnet.predict(X) # Get the final assignments to clusters

y = y_train.detach().cpu().numpy() # In case your labels are of torch.Tensor type.
acc_score = Metrics.acc_score(cluster_assignments, y, n_clusters=2)
nmi_score = Metrics.nmi_score(cluster_assignments, y)
print(f"ACC: {np.round(acc_score, 3)}")
print(f"NMI: {np.round(nmi_score, 3)}")
```

You can read the code docs for more information and functionalities

## Citation

```
@inproceedings{shaham2018,
  author = {Uri Shaham and Kelly Stanton and Henri Li and Boaz Nadler and Ronen Basri and Yuval Kluger},
  title = {SpectralNet: Spectral Clustering Using Deep Neural Networks},
  booktitle = {Proc. ICLR 2018},
  year = {2018}
}
```
