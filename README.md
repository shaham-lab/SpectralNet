# SpectralNet

<p align="center">
    <img src="https://github.com/AmitaiYacobi/SpectralNet01/blob/main/figures/twomoons.png"
</p>


SpectralNet is a python library that performs spectral clustering with deep neural networks.<br><br>
Link to the paper - [SpectralNet](https://openreview.net/pdf?id=HJ_aoCyRZ)

## Requirements

To run SpectralNet, you'll need Python 3.x and to run the requirements.txt file as follows: <br>

```bash
pip3 install -r req.txt
```

You will also need wget to download the Reuters dataset, which for MacOS can be installed with

```bash
brew install wget
```

## Downloading and preprocessing reuters

To run SpectralNet on the Reuters dataset, you must first download and preprocess it. This can be done by

```bash
cd path_to_spectralnet/data/Reuters/; ./get_data.sh; python3 make_reuters.py
```

## Usage

To use SpectralNet on MNIST, Reuters, or the twomoon dataset, `cd` to src directory and run the following:

```bash
python3 main.py config/mnist|reuters|twomoons.json
```
If you want to use your own dataset, you should provide a json config file that looks like the following:

```json
{
    "dataset": {
        "dpath": "path_to_data/name_of_data_file.csv",
        "lpath": "path_to_data/name_of_labels_file.csv" 
    },
    "n_clusters": 3,                            
    "is_sparse_graph": false,                   
    "should_use_ae": false,                 
    "should_use_siamese": false,            
    "should_check_generalization": false,  
    "ae": {
        "architecture": {
            "hidden_dim1": 512,
            "hidden_dim2": 512,
            "hidden_dim3": 2048,
            "output_dim": 10
        },
        "epochs": 30,
        "n_samples": 70000,
        "lr": 1e-3,
        "lr_decay": 0.1,
        "min_lr": 1e-7,
        "patience": 5,
        "batch_size": 256
    },
    "siamese": {
        "architecture": {
            "n_layers": 5,
            "hidden_dim1": 1024,
            "hidden_dim2": 1024,
            "hidden_dim3": 512,
            "output_dim": 10
        },
        "epochs": 100,
        "n_samples": 70000,
        "lr": 1e-3,
        "lr_decay": 0.1,
        "min_lr": 1e-7,
        "patience": 10,
        "n_neighbors": 2,               
        "use_approx": false,            
        "batch_size": 128
    },
    "spectral": {
        "architecture": {
            "n_layers": 4,
            "hidden_dim1": 256,
            "hidden_dim2": 128,
            "output_dim": 3
        },
        "epochs": 80,
        "lr": 1e-3,
        "lr_decay": 0.1,
        "min_lr": 1e-8,
        "batch_size": 50,
        "n_neighbors": 23,
        "scale_k": 2,                   
        "is_local_scale": false,       
        "patience": 10
    }
}

```

The label file is optional and may not be provided. <br><br>
Parameters:<br>
`should_use_ae` - Whether you want to use the AE network as part of the training process.<br>
`should_use_siamese` - Whether you want to use the Siamese network as part of the training process. <br>
`should_check_generalization` -  Whether you want to check how good the network can generalize new samples (Out-of-sample-extention). <br>
`n_neighbors` (in the siamese config) - Threshold where, for all k <= siam_k closest neighbors to x_i, (x_i, k) is considered a 'positive' pair by siamese net<br>
`use_approx` - In case you have a large dataset you may want to compute the nearest neighbors with Annoy instead of KNN. <br>
`scale_k`- Scale for the computation of the Gaussian-kernel affinity matrix. <br>
`is_local_scale` - Global scale of local scale. <br>

## OS & Hardware

The code has been tested on the following hardare and OS: <br>

1. Linux CentOS 7.9.2009, GeForce GTX 1080 Ti
2. macOS Monterey (12.15.1), M1 processor
