from typing import Tuple

from torch.utils.data import DataLoader, Dataset
from torch import nn

from config import Config
from stages import *

from data.util import split_weekdays_and_weekends

from prediction_models.dbn import DBN, pre_train_dbn
from prediction_models.kelm import KELM


class BinaryVectorDataset(Dataset):
    def __init__(self, n_samples: int, n_bits: int):
        self.n_samples = n_samples
        self.n_bits = n_bits
        self.format_str = f'{{0:0{self.n_bits}b}}'
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, index: int) \
            -> Tuple[torch.TensorType, torch.TensorType]:
        bin_str = self.format_str.format(index)
        return torch.tensor([float(c) for c in bin_str], dtype=torch.float32), torch.tensor([index,], dtype=torch.float32)


class RandomBinaryVectorDataset(Dataset):
    def __init__(self, n_samples, bits_per_sample):
        self.n_samples = n_samples
        self.bits_per_sample = bits_per_sample

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> torch.TensorType:
        return (torch.rand(self.bits_per_sample) > 0.5).to(dtype=torch.float32)


class SlidingWindowDataset(Dataset):
    def __init__(self, mat: torch.FloatTensor, sliding_window_length: int, stride: int = 1):
        self.mat = mat
        self.sliding_window_length = sliding_window_length
        self.stride = stride

    def __len__(self) -> int:
        return (self.mat.shape[1] - self.sliding_window_length) // self.stride

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        index *= self.stride
        return self.mat[:, index:index + self.sliding_window_length], self.mat[:, index + self.sliding_window_length]


class EncodingToPredictionDataset(Dataset):
    def __init__(self, source_dataset: SlidingWindowDataset, dbn: DBN):
        super().__init__()
        self.source_dataset = source_dataset
        self.dbn = dbn
    
    def __len__(self) -> int:
        return len(self.source_dataset)
    
    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        X, y = self.source_dataset[index]
        return self.dbn(X), y


def split_train_val_test(mat: torch.FloatTensor, train_portion: float, val_portion: float):
    train_val_split_idx = int(len(mat) * train_portion)
    val_test_split_idx = train_val_split_idx + int(len(mat) * val_portion)

    return mat[:train_val_split_idx], mat[train_val_split_idx:val_test_split_idx], mat[val_test_split_idx:]


def get_pre_trained_dbn(config: Config, n_samples: int = 10000, print_each = 5):
    dbn_pre_train_loader = DataLoader(RandomBinaryVectorDataset(
        n_samples, config.time_window_length), batch_size=256)
    dbn = DBN(config.time_window_length, config.dbn_hidden_layer_sizes,
              config.gibbs_sampling_steps).to(config.device)
    pre_train_dbn(dbn, dbn_pre_train_loader, config.device, print_each=print_each)

    return dbn


def fit_kelm_to_dbn(dbn: DBN, dataset: SlidingWindowDataset):
    e2p_dataset = EncodingToPredictionDataset(dataset, dbn)

    kelm_X = []
    kelm_y = []

    for X, y in e2p_dataset:
        kelm_X.append(X)
        kelm_y.append(y)

    kelm_X = torch.concatenate(kelm_X)
    kelm_y = torch.concatenate(kelm_y)[:, None]

    kelm = KELM()
    kelm.fit(kelm_X, kelm_y)

    return kelm


def run():

    mat_c = config.load('mat_c.pt')
    mat_c_wd, mat_c_we = split_weekdays_and_weekends(
        mat_c, config.train_period[0])
