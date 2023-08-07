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
    
    def forward(self, v):
        for rbm in self.rbms:
            v = rbm.prop_forward(v)
        
        return v


def pre_train_dbn(model: DBN, train_loader: torch.utils.data.DataLoader,
                  n_epochs: int=20, learning_rate: float=0.01,
                  print_every: int=10):
    for epoch in range(n_epochs):
        data = [d for d, _ in train_loader]
        for m, rbm in enumerate(model.rbms):
            optimizer = torch.optim.Adam(rbm.parameters(), learning_rate)
            next_data = []
            losses = []
            for sample in data:
                rbm.train(True)
                v, v_gibbs, h = rbm(sample)
                loss = rbm.free_energy(v) - rbm.free_energy(v_gibbs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rbm.train(False)
                next_data.append(h.detach())
                losses.append(loss.detach())
            if epoch % print_every == 0:
                print(f'Epoch {epoch}, Machine {m}:\tLoss: {mean(losses)}')
            data = next_data
        
    return model
    
def train_dbm(model: DBN, train_loader: torch.utils.data.DataLoader,
              n_epochs: int=20, learning_rate: float=0.01, 
              print_every: int=10):
    optimizer = Adam(model.parameters(), learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()

    losses = []
    for epoch in range(n_epochs):
        for data, _ in train_loader:
            pred = model(data)
            loss = loss_fn(pred, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())

        if epoch % print_every == 0:
            print(f'Epoch {epoch}, Loss: {mean(losses)}')

    model.train(False)        
    
    return model
