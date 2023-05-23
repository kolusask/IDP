from .rbm import RBM, train_rbm
from typing import List, Optional

from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn

from numpy import mean


class DBN(nn.Module):
    def __init__(self, input_size : int, output_sizes : List[int], k: int=1):
        super(DBN, self).__init__()
        self.rbms = nn.ModuleList()
        for os in output_sizes:
            self.rbms.append(RBM(input_size, os, k))
            input_size = os
    
    def forward(self, v, layers_to_use : Optional[int] = None):
        if layers_to_use is None:
            layers_to_use = len(self.rbms)
        
        h = v
        for rbm in self.rbms[:layers_to_use]:
            _, v_gibbs = rbm(h)
            h = rbm.visible_to_hidden(v_gibbs)
        
        return h

    
class PartialDBNDataset(Dataset):
    def __init__(self, base_dataset: Dataset, model: DBN, layers_to_use: int):
        super(PartialDBNDataset, self).__init__()
        self.base_dataset = base_dataset
        self.model = model
        self.layers_to_use = layers_to_use
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, index: int):
        data, _ = self.base_dataset[index]
        asdf = 5
        h = self.model(data, self.layers_to_use)
        return h


def train_dbm(model: DBN, train_loader: torch.utils.data.DataLoader, n_epochs: int=20, n_epochs_per_rbm: int=20, lr: float=0.01, print_every=10, pre_train_rbms=True):
    model.train()
    optimizer = Adam(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()

    if pre_train_rbms:
        for i, rbm in enumerate(model.rbms):
            partial_loader = DataLoader(PartialDBNDataset(train_loader.dataset, model, i))
            train_rbm(rbm, partial_loader, n_epochs_per_rbm, lr, print_every)

    for epoch in range(n_epochs):
        print("DBN epoch " + str(epoch))
        dbn_losses = []
        
        for data, target in train_loader:
            pred = model(data)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dbn_losses.append(loss.detach())
        
        print('Loss=', str(mean(dbn_losses)))

    model.train(False)        
    
    return model
