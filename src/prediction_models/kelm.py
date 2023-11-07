import torch


class KELM:
    def __init__(self):
        self._beta = None

    def fit(self,
            X: torch.TensorType,
            y: torch.TensorType,
            reg_coeff: float = 0.0, gamma: float = 1.0):
        M, N = X.shape
        M, C = y.shape
        assert M == len(X)
        self.gamma = gamma
        omega = self._kernel(X, X)
        if reg_coeff == 0:
            self._beta = torch.pinverse(omega) @ y
        else:
            self._beta = torch.pinverse(torch.eye(M, device=X.device) / reg_coeff + omega) @ y
        self.reg_coeff = reg_coeff
        self._data = X

    def random_fit(self, device, M: int, N: int, C: int, reg_coeff: float = 1.0, gamma: float = 1.0):
        X = torch.rand(M, N, dtype=torch.float32, device=device)
        y = torch.rand(M, C, dtype=torch.float32, device=device)
        self.fit(X, y, reg_coeff=reg_coeff, gamma=gamma)

    def __call__(self, X: torch.TensorType) -> torch.TensorType:
        return self.predict(X)

    def predict(self, X: torch.TensorType) -> torch.TensorType:
        assert self._beta is not None, "KELM model is not fitted"
        M, N = X.shape
        k = self._kernel(X, self._data)
        # if self.reg_coeff > 0:
        #     k += torch.eye(M, device=X.device) / self.reg_coeff
        return k @ self._beta

    def _kernel(self, x1: torch.TensorType, x2: torch.TensorType) \
            -> torch.TensorType:
        return torch.exp(
            -self.gamma * torch.norm(x1.cpu()[:, None] - x2.cpu()[None, :], dim=2) ** 2
        ).to(x1.device)
