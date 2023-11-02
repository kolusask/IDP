import torch

from datetime import datetime

from data.extract import DetectorDataProvider, LookUpTable
from data.compress import *
from data.util import count_points_in_period, crop_q_between

from globals import Period


def extract_data(load_path, start_date: datetime, end_date: datetime, save_path=None):
    lut = LookUpTable(load_path)
    int_det = lut.get_detectors_per_section()
    ddp = DetectorDataProvider(load_path)
    int_det['Counts'] = int_det.apply(
        lambda sec: ddp.get_counts_entering_section(
            sec['End'], sec['Detectors'], start_date, end_date
        ),
        axis=1
    )
    mat_q = torch.tensor(int_det['Counts'].tolist()).T

    return mat_q


def preprocess_data(threshold, mat_q):
    mat_q_fft = torch.fft.fft(mat_q, dim=0)
    mat_q_fft_mag = torch.abs(mat_q_fft)
    mat_q_trend_fft = mat_q_fft#.clone()
    mat_q_trend_fft[mat_q_fft_mag < threshold] = 0.0
    mat_q_trend = torch.fft.ifft(mat_q_trend_fft, dim=0)
    mat_q_resid = mat_q - mat_q_trend

    return mat_q_trend, mat_q_resid


def compress_data(mat_q: torch.Tensor, read_period: Period, train_period: Period, alpha: float):
    mat_r, nonempty = build_correlation_matrix(mat_q, True)
    mat_q = crop_q_between(mat_q, read_period, train_period)
    mat_q = mat_q[:, nonempty]
    groups = split_sections_into_groups(mat_r, alpha)
    mat_c, representatives = get_compression_matrix(mat_q, groups)
    mat_x = get_compressed_matrix(mat_c, mat_q)

    return mat_c, mat_x, representatives