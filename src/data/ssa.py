import torch
from typing import List
from scipy.linalg import hankel


class SSA:
    def __init__(self, L: int, I: int | List[list | torch.Tensor] | torch.Tensor):
        """
        Parameters:
            L (int):                                Embedding dimension
            I (int | list[list | tensor] | tensor): Array with elementary matrices indices
        """
        self.L = L
        self.I = self._format_I(I)

    def _format_I(self, I):
        if type(I) is int:
            I = torch.linspace(0, self.L, I + 1, dtype=int)
            I = [torch.arange(I[i], I[i + 1]) for i in range(len(I) - 1)]
        elif type(I) is list:
            used_groups = torch.zeros(self.L, dtype=bool)
            for i in I:
                assert len(i.shape) == 1 and sum(used_groups[i]) == 0
                used_groups[i] = True
        elif type(I) is torch.tensor:
            assert (len(I.shape) == 1 and I.shape <= (self.L,))\
                or (len(I.shape) == 2 and I.shape[0] + I.shape[1] <= self.L)

        return I
    
    def _embed(self, x):
        return torch.from_numpy(hankel(x[:self.L], x[self.L - 1:]))
    
    def _svd(m):
        return torch.linalg.svd(m, full_matrices=False)
    
    def _to_sequences(self, U, S, Vh):
        elem_mat = torch.stack(
            [S[i] * (torch.outer(U[:, i], Vh[i])) for i in range(self.L)])
        grouped_mat = [torch.sum(elem_mat[g], dim=0) for g in self.I]
        for mat in grouped_mat:
            h, w = mat.shape
            for i in range(0, w + h):
                v_ind = torch.arange(max(0, i - w + 1), min(i + 1, h))
                h_ind = torch.arange(min(i, w - 1), max(-1, i - h), -1)
                mat[v_ind, h_ind] = mat[v_ind, h_ind].mean()

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
        N = len(x)  # length of time-series
        y = torch.zeros((len(self.I), N))
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

        tau = self._embed(x)
        U, S, Vh = SSA._svd(tau)
        y = torch.zeros(len(self.I), len(x) + M)
        y[:, :len(x)] = self._to_sequences(U, S, Vh)
        P = U.T[:, :self.L - 1]
        pi = U.T[:, self.L - 1]
        for i, g in enumerate(self.I):
            pi_g = pi[g]
            nu_2 = pi_g @ pi_g
            R = 1 / (1 - nu_2) * (pi_g @ P[g])
            for m in range(len(x), len(x) + M):
                y[i, m] = R @ y[i, m - len(R):m]
        
        return y