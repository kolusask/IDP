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
        self.gamma = gamma
        omega = self._kernel(X, X)
        # self._beta = torch.inverse(torch.eye(M, device=X.device) / reg_coeff + omega) @ y
        self._beta = torch.pinverse(omega) @ y
        self._data = X

    def random_fit(self, device, M: int, N: int, C: int, reg_coeff: float = 1.0, gamma: float = 1.0):
        X = torch.rand(M, N, dtype=torch.float32, device=device)
        y = torch.rand(M, C, dtype=torch.float32, device=device)
        self.fit(X, y, reg_coeff=reg_coeff, gamma=gamma)

    def __call__(self, X: torch.TensorType) -> torch.TensorType:
        return self.predict(X)

    def predict(self, X: torch.TensorType) -> torch.TensorType:
        assert self._beta is not None, "The model is not fitted"
        return self._kernel(X, self._data) @ self._beta

    def _kernel(self, x1: torch.TensorType, x2: torch.TensorType) \
            -> torch.TensorType:
        return torch.exp(
            -self.gamma * torch.norm(x1[:, None] - x2[None, :], dim=2) ** 2
        )


# class DatasetKELM(KELM):
#     def fit(self, dataset: torch.utils.data.Dataset, encoder: torch.nn.Module,
#             reg_coeff: float = 1.0, gamma: float = 1.0):
#         self.dataset = dataset
#         self.encoder = encoder
#         self.reg_coeff = reg_coeff
#         self.gamma = gamma
#         data_X = []
#         data_y = []
#         for i in range(len(dataset)):
#             X, y = dataset[i]
#             data_X.append(X)
#             data_y.append(y)
#         data_X = torch.stack(data_X)
#         data_y = torch.stack(data_y)
#         self.beta = []
#         for X, y in zip(data_X, data_y):
#             omega = self._kernel(X[None], data_X).squeeze()
#             self.beta.append((omega @ data_y).item())
#         self.beta = torch.tensor(self.beta)
#         pass

#     def predict(self, X: torch.FloatTensor) -> torch.FloatTensor:

        
#     def _kernel(self, X):
#         row = []
#         for i in range(len(self.dataset)):
#             data_X, y = self.dataset[i]
#             omega = super()._kernel(X[None], data_X).squeeze()
#             row.append((omega @ y).item())

#         return torch.tensor(row)

