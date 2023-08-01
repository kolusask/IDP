import torch

from datetime import datetime
from scipy.linalg import hankel

POINTS_PER_HOUR = 60 // 15
POINTS_PER_DAY = POINTS_PER_HOUR * 24


def count_points_between(start_date: datetime, end_date: datetime):
    return int((end_date - start_date).total_seconds()
               // 3600 * POINTS_PER_HOUR)


def crop_q_between(mat_q, old_start, old_end, new_start, new_end):
    assert old_start <= new_start < new_end <= old_end
    beg_offset = count_points_between(old_start, new_start)
    end_offset = count_points_between(new_end, old_end)

    return mat_q[beg_offset:len(mat_q) if end_offset == 0 else -end_offset]


def extract_week_day(mat, start_date: datetime, weekday: int):
    assert len(mat) % POINTS_PER_DAY == 0
    return mat[
        (torch.arange(len(mat) // POINTS_PER_DAY) + start_date.weekday())
        .repeat_interleave(POINTS_PER_DAY) % 7 == weekday
    ]


def split_weekdays_and_weekends(mat, start_date: datetime, end_date: datetime):
    assert start_date.hour == 0\
        and start_date.minute == 0\
        and start_date.second == 0\
        and start_date.microsecond == 0
    assert end_date.hour == 0\
        and end_date.minute == 0\
        and end_date.second == 0\
        and end_date.microsecond == 0

    weekdays = torch.stack([extract_week_day(mat, start_date, d)
                           for d in range(5)]).t().flatten(0, 0)
    weekends = torch.stack([extract_week_day(mat, start_date, d)
                           for d in range(5, 7)]).t().flatten(0, 0)

    return weekdays, weekends


def ssa_format_I(L, I):
    if type(I) is int:
        I = torch.linspace(0, L, I + 1, dtype=int)
        I = [torch.arange(I[i], I[i + 1]) for i in range(len(I) - 1)]
    elif type(I) is list:
        used_groups = torch.zeros(L, dtype=bool)
        for i in I:
            assert len(i.shape) == 1 and sum(used_groups[i]) == 0
            used_groups[i] = True
    elif type(I) is torch.tensor:
        assert (len(I.shape) == 1 and I.shape <= (L,))\
            or (len(I.shape) == 2 and I.shape[0] + I.shape[1] <= L)

    return I


def ssa(x, L, I=3):
    I = ssa_format_I(L, I)
    tau = torch.from_numpy(hankel(x[:L], x[L - 1:]))
    U, S, Vh = torch.linalg.svd(tau, full_matrices=False)
    elem_mat = torch.stack(
        [S[i] * (torch.outer(U[:, i], Vh[i])) for i in range(L)])
    grouped_mat = [torch.sum(elem_mat[g], dim=0) for g in I]
    for mat in grouped_mat:
        h, w = mat.shape
        for i in range(0, w + h):
            v_ind = torch.arange(max(0, i - w + 1), min(i + 1, h))
            h_ind = torch.arange(min(i, w - 1), max(-1, i - h), -1)
            mat[v_ind, h_ind] = mat[v_ind, h_ind].mean()
    return torch.stack([torch.concat([mat[:-1, 0], mat[-1, :]]) for mat in grouped_mat])


def ov_ssa(x, L, I, Z, q):
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
        L (int):      Embedding dimension
        I (array):    Array with elementary matrices indices
        Z (int):      Local segment length
        q (int):      The number of samples that are reconstructed locally

    Returns:
        y (array): Reconstructed SSA time-series

    """
    # Parameters of the algorithm
    I = ssa_format_I(L, I)
    N = len(x)  # length of time-series
    y = torch.zeros((len(I), N))
    L_B = (Z - q) / 2  # number of points discarded at each iteration
    P = (N - Z) / q + 1  # the number of iterations

    # L_B must be an integer, otherwise ValueError is raised
    if int(L_B) != L_B:
        raise ValueError(L_B)
    L_B = int(L_B)
    P = int(P)

    # First iteration
    series = x[:Z]
    y_aux = ssa(series, L, I)
    y[:, :Z - L_B] = y_aux[:, :Z - L_B]

    # Loop
    for p in range(1, P):
        rho = (p - 1) * q
        series = x[rho:rho + Z]
        y_aux = ssa(series, L, I)
        y[:, rho + L_B:rho + L_B + q] = y_aux[:, L_B:L_B + q]

    # Last iteration
    rho = (P - 1) * q
    series = x[rho:]
    y_aux = ssa(series, L, I)
    y[:, rho + L_B:] = y_aux[:, L_B:]

    return y
