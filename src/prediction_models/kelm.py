import torch

def gaussian_kernel(x1: torch.TensorType, x2: torch.TensorType, gamma: float) \
        -> torch.TensorType:
    return torch.exp(-gamma * torch.cdist(x1, x2).pow(2))

class KELM:
    def __init__(self):
        self._beta = None

    def fit(self,
            X: torch.TensorType,
            y: torch.TensorType,
            reg_coeff: float=1.0, gamma: float=1.0, clean: bool=True):
        omega = gaussian_kernel(X, X, gamma)
        self._beta = \
            torch.inverse(torch.eye(X.shape[0]) / reg_coeff + omega) @ y
        self._data = X
        self._gamma = gamma
    
    def __call__(self, X: torch.TensorType) -> torch.TensorType:
        return self.predict(X)

    def predict(self, X: torch.TensorType) -> torch.TensorType:
        assert self._beta is not None, "The model is not fitted"
        return gaussian_kernel(X, self._data, self._gamma) * self._beta