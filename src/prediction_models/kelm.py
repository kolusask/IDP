import torch


class KELM:
    def __init__(self):
        self._beta = None

    def fit(self,
            X: torch.TensorType,
            y: torch.TensorType,
            reg_coeff: float = 1.0, gamma: float = 1.0):
        M, N = X.shape
        M, C = y.shape
        assert M == len(X)
        self._gamma = gamma
        omega = self._kernel(X, X)
        self._beta = torch.inverse(torch.eye(M) / reg_coeff + omega) @ y
        self._data = X

    def random_fit(self, M: int, N: int, C: int, reg_coeff: float = 1.0, gamma: float = 1.0):
        X = torch.rand(M, N, dtype=torch.float32)
        y = torch.rand(M, C, dtype=torch.float32)
        self.fit(X, y, reg_coeff=reg_coeff, gamma=gamma)

    def __call__(self, X: torch.TensorType) -> torch.TensorType:
        return self.predict(X)

    def predict(self, X: torch.TensorType) -> torch.TensorType:
        assert self._beta is not None, "The model is not fitted"
        return self._kernel(X, self._data) @ self._beta

    def _kernel(self, x1: torch.TensorType, x2: torch.TensorType) \
            -> torch.TensorType:
        return torch.exp(
            -self._gamma * torch.norm(x1[:, None] - x2[None, :], dim=2) ** 2
        )
