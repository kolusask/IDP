import torch
from typing import List


class SSA:
    def __init__(self, L: int, I: int | List[list | torch.Tensor] | torch.Tensor, device: torch.device):
        """
        Parameters:
            L (int):                                Embedding dimension
            I (int | list[list | tensor] | tensor): Array with elementary matrices indices
        """
        self.device = device
        self.L = L
        self.I = self._format_I(I)

    def _format_I(self, I):
        if type(I) is int:
            I = torch.linspace(0, self.L, I + 1, dtype=int)
            I = [torch.arange(I[i], I[i + 1]) for i in range(len(I) - 1)]
        elif type(I) is list:
            used_groups = torch.zeros(self.L, dtype=bool)
            list_type = None
            tensor_list = []
            for i in I:
                assert list_type is None or type(i) is list_type
                list_type = type(i)
                if list_type is int:
                    i = torch.arange(sum(used_groups), sum(used_groups) + i, dtype=int)
                assert len(i.shape) == 1 and sum(used_groups[i]) == 0
                tensor_list.append(i)
                used_groups[i] = True
            I = tensor_list
        elif type(I) is torch.tensor:
            assert (len(I.shape) == 1 and I.shape <= (self.L,))\
                or (len(I.shape) == 2 and I.shape[0] + I.shape[1] <= self.L)

        return I
    
    def _embed(self, x):
        N, B = x.shape
        hankel = torch.zeros(B, self.L, N - self.L + 1).to(self.device)
        hankel[:, :, 0] = x[:self.L].T
        hankel[:, -1] = x[self.L - 1:].T
        for r in range(self.L - 2, -1, -1):
            hankel[:, r, 1:] = hankel[:, r + 1][:, :-1]
        
        return hankel
    
    def _svd(m):
        return torch.linalg.svd(m, full_matrices=False)
    
    def _to_sequences(self, U, S, Vh):
        # for i in range(self.L):
        #     a = torch.outer(U[0][:, i], Vh[0][i])
        #     b = U[:, :, i][..., None, :] * Vh[:, i][..., None]
        #     pass
        # elem_mat = torch.stack(
        #     [S[i] * (torch.outer(U[:, i], Vh[i])) for i in range(self.L)])
        # zero_mat = torch.stack(
        #     [S[0][i] * (torch.outer(U[0][:, i], Vh[0][i])) for i in range(self.L)])
        elem_mat = torch.stack(
            [S[:, i] * (U[:, :, i][..., None, :] * Vh[:, i][..., None]).permute(2, 1, 0) for i in range(self.L)])
        grouped_mat = [torch.sum(elem_mat[g], dim=0) for g in self.I]
        for mat in grouped_mat:
            h, w, b = mat.shape
            for i in range(0, w + h):
                v_ind = torch.arange(max(0, i - w + 1), min(i + 1, h))
                h_ind = torch.arange(min(i, w - 1), max(-1, i - h), -1)
                mat[v_ind, h_ind] = mat[v_ind, h_ind].mean(dim=0)

        return torch.stack([torch.concat([mat[:-1, 0], mat[-1, :]]) for mat in grouped_mat])

    def ssa(self, x):
        tau = self._embed(x)
        U, S, Vh = SSA._svd(tau)
        return self._to_sequences(U, S, Vh)

    def ov_ssa(self, x, Z, q):
        """
        https://www.sciencedirect.com/science/article/pii/S2352711017300596#b1

        Consider a time series segment of length Z. 
        All samples within this segment are used to compute the SSA locally. 
        However, since the SSA method suffers from boundary effects, 
        the extreme points in the left and right edges are discarded. 
        The quantity of discarded samples is given by L_B = (Z-q)/2. 
        Only an inner subset of samples, q, is considered meaningful to represent the local time-series Z. 
        The final reconstruction is given by the concatenation of the inner segmentes q, which do not overlap. 
        The extreme edges of the original time-series need special attention.
        In the first run only the last L_B points are discarded.
        On the other hand, in the last run, the first L_B points are discarded.
        This approach is a modification of  the overlap-save method, 
        a classic tool to calculate FFT (Fast Fourier Transform) convolution of 
        infinite (and finite) duration time series. 
        This adaptation was necessary because the standard SSA algorithm suffers 
        from boundary effects on both sides. In the overlap-save method only the initial 
        points must be discarded, because boundary effects occur only at the filtering initialization. 
        For a complete discussion about the method, see Leles et al (2017) A New
        Algorithm in Singular Spectrum Analysis Framework: the Overlap-SSA (ov-SSA). 

        Parameters:
            x (array):    Original time-series
            Z (int):      Local segment length
            q (int):      The number of samples that are reconstructed locally

        Returns:
            y (array): Reconstructed SSA time-series

        """
        # Parameters of the algorithm
        N, B = x.shape  # length of time-series
        y = torch.zeros((len(self.I), N, B))
        L_B = (Z - q) / 2  # number of points discarded at each iteration
        P = (N - Z) / q + 1  # the number of iterations

        # L_B must be an integer, otherwise ValueError is raised
        if int(L_B) != L_B:
            raise ValueError(L_B)
        L_B = int(L_B)
        P = int(P)

        # First iteration
        series = x[:Z]
        y_aux = self.ssa(series)
        y[:, :Z - L_B] = y_aux[:, :Z - L_B]

        # Loop
        for p in range(1, P):
            rho = (p - 1) * q
            series = x[rho:rho + Z]
            y_aux = self.ssa(series)
            y[:, rho + L_B:rho + L_B + q] = y_aux[:, L_B:L_B + q]

        # Last iteration
        rho = (P - 1) * q
        series = x[rho:]
        y_aux = self.ssa(series)
        y[:, rho + L_B:] = y_aux[:, L_B:]

        return y
    
    def forecast(self, x, M):
        # https://www.researchgate.net/publication/228092069_Basic_Singular_Spectrum_Analysis_and_Forecasting_with_R

        N, B = x.shape
        T = len(self.I)
        empty_input = x.sum(dim=0) == 0
        tau = self._embed(x)
        U, S, Vh = SSA._svd(tau)
        del tau
        rec = self._to_sequences(U, S, Vh)
        y = torch.concat([rec, torch.zeros(T, M, B).to(self.device)], dim=1).permute(2, 0, 1).to(self.device)
        P = U.T[:, :-1].permute(2, 0, 1)
        pi = U.T[:, -1].T
        for i, g in enumerate(self.I):
            pi_g = pi[:, None, g]
            nu_2 = torch.bmm(pi_g.view(B, 1, len(g)), pi_g.view(B, len(g), 1))
            R = (1 / (1 - nu_2) * (torch.bmm(pi_g, P[:, g]))).squeeze(1)
            for m in range(len(x), len(x) + M):
                y[~empty_input, i, m] = torch.bmm(R[:, None], y[:, i, m - R.shape[1]:m][:, :, None]).squeeze([1, 2])[~empty_input]
        
        return y.permute(1, 2, 0)


class SequentialSSA:
    def __init__(self,
                 L_trend: int, I: int | List[list | torch.Tensor] | torch.Tensor):
        self.trend_ssa = SSA(L_trend, 2)
        self.I = I

    def ssa(self, x):
        return self._extract_trend(SSA.ssa, x)

    def ov_ssa(self, x, Z, q):
        # TODO: Extract
        return self.trend_ssa.ov_ssa(x, Z, q)

    def forecast(self, x, M):
        return self._extract_trend(SSA.forecast, x, M)

    def _extract_trend(self, fn, x, *args):
        trend_resid = fn(self.trend_ssa, x, *args)
        trend = trend_resid[0]
        resid = trend_resid[1]
        del trend_resid
        L = len(x) // 2
        resid = fn(SSA(L, self.I), resid[:len(x)], *args)

        return torch.row_stack([trend.unsqueeze(0), resid])
