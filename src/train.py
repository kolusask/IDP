import torchmetrics as tm

from tqdm import tqdm

from itertools import accumulate
from typing import List, Tuple

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
    def __init__(self, mat_list: List[torch.FloatTensor], sliding_window_length: int, stride: int = 1):
        assert len(mat_list) > 0
        self.mat_list = mat_list
        self.sliding_window_length = sliding_window_length
        self.stride = stride
        mat_lengths = [(mat.shape[1] - self.sliding_window_length) // self.stride for mat in mat_list]
        self.mat_len_acc = [0,] + [int(i) for i in accumulate(mat_lengths)]

    def __len__(self) -> int:
        return self.mat_len_acc[-1]
    
    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        mat_idx = next(i for i, l in enumerate(self.mat_len_acc) if l > index) - 1
        index -= self.mat_len_acc[mat_idx]
        mat = self.mat_list[mat_idx]
        return mat[:, index:index + self.sliding_window_length], mat[:, index + self.sliding_window_length]


class EncodingToPredictionDataset(Dataset):
    def __init__(self, source_dataset: SlidingWindowDataset, dbn: DBN, precompute: bool=False):
        super().__init__()

        self.cache = None
        if precompute:
            self.cache = []
            for i in range(len(source_dataset)):
                X, y = source_dataset[i]
                self.cache.append((dbn(X), y))
        else:
            self.source_dataset = source_dataset
            self.dbn = dbn

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if self.cache is None:
            X, y = self.source_dataset[index]
            return self.dbn(X), y
        else:
            return self.cache[index]


def split_train_val_test(mat_list: List[torch.FloatTensor], train_portion: float, val_portion: float):
    mat_len = mat_list[0].shape[1]
    train_val_split_idx = int(mat_len * train_portion)
    val_test_split_idx = train_val_split_idx + int(mat_len * val_portion)

    train_list = []
    val_list = []
    test_list = []

    for mat in mat_list:
        train_list.append(mat[:, :train_val_split_idx])
        val_list.append(mat[:, train_val_split_idx:val_test_split_idx])
        test_list.append(mat[:, val_test_split_idx:])
    
    return train_list, val_list, test_list


def get_pre_trained_dbn(config: Config, n_samples: int = 10000, print_each=5):
    dbn_pre_train_loader = DataLoader(RandomBinaryVectorDataset(
        n_samples, config.time_window_length), batch_size=256)
    dbn = DBN(config.time_window_length, config.dbn_hidden_layer_sizes,
              config.gibbs_sampling_steps).to(config.device)
    pre_train_dbn(dbn, dbn_pre_train_loader,
                  config.device, print_each=print_each)

    return dbn


def errors_for_kelm_parameters(X, y, gamma, reg_coeff):
    kelm = KELM()
    kelm.fit(X, y, gamma=gamma, reg_coeff=reg_coeff)

    return torch.abs(kelm(X) - y)


def fit_kelm_to_dbn(dbn: DBN, dataset: SlidingWindowDataset, gamma=1, reg_coeff=1.0):
    e2p_dataset = EncodingToPredictionDataset(dataset, dbn)

    kelm_X = []
    kelm_y = []

    for X, y in e2p_dataset:
        kelm_X.append(X)
        kelm_y.append(y)

    kelm_X = torch.concatenate(kelm_X)
    kelm_y = torch.concatenate(kelm_y)[:, None]

    kelm = KELM()
    kelm.fit(kelm_X, kelm_y, gamma=gamma, reg_coeff=reg_coeff)

    return kelm


def epoch(dbn: DBN, kelm: KELM, dataloader: DataLoader, loss_fn: tm.Metric, device: torch.DeviceObjType) -> torch.FloatTensor:
    n_samples = 0
    loss = torch.tensor([0.,]).to(device)
    for X, y in dataloader:
        n_samples += 1
        pred = dbn(X).squeeze()
        pred = kelm(pred).T

        loss += loss_fn(pred, y)

    return loss / n_samples


def get_datasets(mat_list, config: Config):
    split = split_train_val_test(mat_list, *config.data_split)
    return [SlidingWindowDataset(part, config.time_window_length) for part in split]


def split_mat(mat, config: Config):
    mat_list_wd, mat_list_we = split_weekdays_and_weekends(mat, config.train_period[0])
    return get_datasets(mat_list_wd, config), get_datasets(mat_list_we, config)


def crop_and_split_mat(mat, config: Config):
    mat = crop_q_between(mat, config.read_period, config.train_period)
    dataset_lists = split_mat(mat, config)
    return [[DataLoader(dataset) for dataset in dataset_list] for dataset_list in dataset_lists]


def train_with_config(config: Config, dbn: DBN, datasets: List[Dataset], dbn_training_epochs: int = 0, dbn_eval_each: int = 10, stride=1, gamma=1, reg_coeff=1):
    mse = tm.MeanSquaredError().to(config.device)

    train_dataset, val_dataset, test_dataset = datasets

    kelm = fit_kelm_to_dbn(dbn, train_dataset, gamma=gamma, reg_coeff=reg_coeff)
    # kelm = None

    if dbn_training_epochs > 0:
        train_dataloader = DataLoader(train_dataset)
        val_dataloader = DataLoader(val_dataset)

        optim = torch.optim.Adam(dbn.parameters())
        best_loss = epoch(dbn, kelm, val_dataloader, mse, config.device)
        best_state_dict = dbn.state_dict()

        dbn.train(True)
        for dbn_epoch in tqdm(range(dbn_training_epochs)):
            optim.zero_grad()
            train_loss: torch.FloatTensor = epoch(
                dbn, kelm, train_dataloader, mse, config.device)
            train_loss.backward()

            optim.step()
            kelm = fit_kelm_to_dbn(dbn, train_dataset, gamma=gamma)

            if dbn_epoch % dbn_eval_each == 0:
                dbn.train(False)
                val_loss = epoch(dbn, kelm, val_dataloader, mse, config.device)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state_dict = dbn.state_dict()
                dbn.train(True)
        dbn.train(False)

        dbn.load_state_dict(best_state_dict)

    return dbn, kelm, val_dataloader
