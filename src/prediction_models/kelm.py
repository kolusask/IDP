import torch


class KELM:
    def __init__(self):
        self._beta = None

    def fit(self,
            X: torch.TensorType,
            y: torch.TensorType,
            reg_coeff: float = 1.0, gamma: float = 1.0, clean: bool = True):
        self._gamma = gamma
        X_tile = torch.tile(X, (1, X.shape[1])).reshape(*X.shape, -1)
        omega = torch.exp(-gamma * (X_tile - X_tile.swapaxes(1, 2)).pow(2))
        self._beta = \
            torch.bmm(
                torch.inverse((torch.eye(X.shape[1]) / reg_coeff) + omega),
                y[..., None]
                )
        self._data = X

    def __call__(self, X: torch.TensorType) -> torch.TensorType:
        return self.predict(X)

    def predict(self, X: torch.TensorType) -> torch.TensorType:
        assert self._beta is not None, "The model is not fitted"
        return self._kernel(X, self._data) * self._beta

    def _kernel(self, x1: torch.TensorType, x2: torch.TensorType) \
            -> torch.TensorType:
        return torch.exp(-self._gamma * torch.norm(x1 - x2).pow(2))
