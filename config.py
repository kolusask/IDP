
import os
import sys
import torch

from datetime import datetime
from json import load
from typing import List
from warnings import warn

from globals import Period


if 'src' not in sys.path:
    sys.path.append('src')


class Config:
    DATE_FORMAT = '%Y-%m-%d'

    def __init__(self, name: str,
                 read_period: Period,
                 train_period: Period,
                 spectral_threshold: int,
                 alpha: int,
                 dbn_hidden_layer_sizes: List[int] | int,
                 gibbs_sampling_steps: int,
                 time_window_length: int,
                 data_split: List[float]):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            warn('CUDA is not available')
            self.device = torch.device('cpu')

        self.outputs_folder = os.path.join('outputs', name)
        if not os.path.exists(self.outputs_folder):
            os.makedirs(self.outputs_folder)

        assert read_period[0] <= train_period[0] < train_period[1] <= read_period[1]
        self.read_period = read_period
        self.train_period = train_period
        self.spectral_threshold = spectral_threshold
        self.alpha = alpha
        if type(dbn_hidden_layer_sizes) is int:
            dbn_hidden_layer_sizes = [dbn_hidden_layer_sizes,] * 3
        self.dbn_hidden_layer_sizes = dbn_hidden_layer_sizes
        self.gibbs_sampling_steps = gibbs_sampling_steps
        self.time_window_length = time_window_length
        assert 0. < data_split[0] < 1.
        assert 0. < data_split[1] < 1.
        # assert sum(data_split) < 1.
        self.data_split = data_split

    def from_json(config_file_path: str) -> 'Config':
        with open('config.json') as config:
            json = load(config)

        return Config(
            json['CONFIG_NAME'],
            (
                Config._parse_date(json['READ_START_DATE']),
                Config._parse_date(json['READ_END_DATE']),
            ),
            (
                Config._parse_date(json['TRAIN_START_DATE']),
                Config._parse_date(json['TRAIN_END_DATE']),
            ),
            json['SPECTRAL_THRESHOLD'],
            json['ALPHA'],
            json['DBN_HIDDEN_LAYER_SIZES'],
            json['GIBBS_SAMPLING_STEPS'],
            json['TIME_WINDOW_LENGTH'],
            json['DATA_SPLIT']
        )

    def _parse_date(date_str: str):
        return datetime.strptime(date_str, Config.DATE_FORMAT)

    def out_path(self, p: str):
        return os.path.join(self.outputs_folder, p)
    
    def save(self, obj: torch.Tensor, path: str):
        torch.save(obj, self.out_path(path))
    
    def load(self, path: str) -> torch.Tensor:
        return torch.load(self.out_path(path), map_location=self.device)


CONFIG = Config.from_json('config.json')
