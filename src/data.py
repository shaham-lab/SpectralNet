import h5py
import torch
import numpy as np

from sklearn.datasets import make_moons
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def load_mnist() -> tuple:
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root='../data',
                                train=True,
                                download=True,
                                transform=tensor_transform)
    test_set = datasets.MNIST(root='../data',
                                train=False,
                                download=True,
                                transform=tensor_transform)
    
    x_train, y_train = zip(*train_set)
    x_train, y_train = torch.cat(x_train), torch.Tensor(y_train)
    x_test, y_test = zip(*test_set)
    x_test, y_test = torch.cat(x_test), torch.Tensor(y_test)

    return x_train, y_train, x_test, y_test

def load_twomoon() -> tuple:
        data, y = make_moons(n_samples=7000, shuffle=True, noise=0.075, random_state=None)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)
        x_train, x_test = torch.Tensor(x_train), torch.Tensor(x_test)
        y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
        return x_train, y_train, x_test, y_test

def load_reuters() -> tuple:
    with h5py.File('../data/Reuters/reutersidf_total.h5', 'r') as f:
        x = np.asarray(f.get('data'), dtype='float32')
        y = np.asarray(f.get('labels'), dtype='float32')

        n_train = int(0.9 * len(x))
        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test

def load_from_path(dpath: str, lpath: str = None) -> tuple:
    X = np.loadtxt(dpath, delimiter=',', dtype=np.float32)
    n_train = int(0.9 * len(X))

    x_train, x_test = X[:n_train], X[n_train:]
    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    
    if lpath is not None:
        y = np.loadtxt(lpath, delimiter=',', dtype=np.float32)
        y_train, y_test = y[:n_train], y[n_train:]
        y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    else:
        y_train, y_test = None, None
    

    return x_train, y_train, x_test, y_test

def load_data(dataset: str) -> tuple:
    """
    This function loads the dataset specified in the config file.
    

    Args:
        dataset (str or dictionary):    In case you want to load your own dataset, 
                                        you should specify the path to the data (and label if applicable) 
                                        files in the config file in a dictionary fashion under the key "dataset".

    Raises:
        ValueError: If the dataset is not found in the config file.

    Returns:
        tuple: A tuple containing the train and test data and labels.
    """

    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist()
    elif dataset == 'twomoons':
        x_train, y_train, x_test, y_test = load_twomoon()
    elif dataset == 'reuters':
        x_train, y_train, x_test, y_test = load_reuters()
    else:
        try:
            data_path = dataset["dpath"]
            if "lpath" in dataset:
                label_path = dataset["lpath"]
            else:
                label_path = None
        except:
            raise ValueError("Could not find dataset path. Check your config file.")
        x_train, y_train, x_test, y_test = load_from_path(data_path, label_path)
        
    return x_train, x_test, y_train, y_test