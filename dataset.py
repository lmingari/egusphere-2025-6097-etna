import numpy as np
import torch
from torch.utils.data import Dataset

#######################
### Transformations ###
#######################
class Standardize:
    def __init__(self, mean, std, eps=1e-6):
        self.mean = torch.from_numpy(mean).float()
        self.std  = torch.from_numpy(std).float()
        self.eps  = eps

    def __call__(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def invert(self, x):
        return x * (self.std + self.eps) + self.mean

class MinMaxScale:
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

    def __call__(self, x):
        return (x - self.min) / (self.max - self.min)

    def invert(self, x):
        return x * (self.max - self.min) + self.min

class LogMinMaxScale:
    """
    Log1p transform followed by a fixed linear rescale.

    Forward:  y = (log1p(x) - min_value) / (max_value - min_value)
    Inverse:  x = expm1(y * (max_value - min_value) + min_value)

    Args:
        min_value (float): lower bound of the *log-space* linear rescale
        max_value (float): upper bound of the *log-space* linear rescale
    """
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        self.range     = max_value - min_value

    def __call__(self, x):
        return self.apply(x)

    def apply(self, x):
        x_log = self._log1p(1E6*x)
        return (x_log - self.min_value) / self.range

    def invert(self, y):
        x_log = y * self.range + self.min_value
        return self._expm1(x_log)*1E-6

    @staticmethod
    def _log1p(x):
        if isinstance(x, torch.Tensor):
            return torch.log1p(x)
        return np.log1p(x)

    @staticmethod
    def _expm1(x):
        if isinstance(x, torch.Tensor):
            return torch.expm1(x)
        return np.expm1(x)

################
### Datasets ###
################

# Dataset for deposit thickness
class ThicknessDataset(Dataset):
    def __init__(self, X, y, transform = None):
        """
        X: numpy array (N, 2) with [lat, lon]
        y: numpy array (N,)
        """
        self.x = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

# Dataset for ensemble forecasts
class EnsembleDataset(Dataset):
    def __init__(self, data_array, transform = None):
        """
        Parameters
        ----------
        data_array : xr.DataArray
            Input 3D xarray DataArray with dimensions (ens, lat, lon), where:
            - ens: ensemble members (e.g., different model runs or simulations)
            - lat: latitude coordinates
            - lon: longitude coordinates
        """
        self.X = data_array.values
        self.transform = transform
        
    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, idx):
        x = self.X[idx]
        x = torch.as_tensor(x, dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        x = x.unsqueeze(0)  # add channel dimension
        return x
