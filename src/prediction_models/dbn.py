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
        self.rbms: List[RBM] = nn.ModuleList()
        for os in output_sizes:
            self.rbms.append(RBM(input_size, os, k))
            input_size = os
    
    def forward(self, v) -> torch.FloatTensor:
        for rbm in self.rbms:
            v = rbm.prop_forward(v)
        
        return v


def pre_train_dbn(model: DBN, train_loader: torch.utils.data.DataLoader,
                  device, n_epochs: int=20, learning_rate: float=0.01,
                  print_each: int=10):
    for epoch in range(n_epochs):
        epoch_losses = torch.zeros(len(model.rbms))
        n_data_points = 0
        for data in train_loader:
            n_data_points += 1
            data = data.to(device)
            for m, rbm in enumerate(model.rbms):
                optimizer = torch.optim.Adam(rbm.parameters(), learning_rate)

                rbm.train(True)
                v, v_gibbs, h = rbm(data)
                optimizer.zero_grad()
                loss = rbm.free_energy(v) - rbm.free_energy(v_gibbs)
                loss.backward()
                optimizer.step()
                rbm.train(False)

                data = h.detach()
                epoch_losses[m] += loss.item()
        if print_each > 0 and epoch % print_each == 0:
            print(f'Epoch {epoch}:')
            epoch_losses /= n_data_points
            for m, loss in enumerate(epoch_losses):
                print(f'\tRBM {m}: Loss={loss}')

    return model
    

def train_dbn(model: DBN, train_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, n_epochs: int=20,
              learning_rate: float=0.01, print_every: int=10):
    optimizer = Adam(model.parameters(), learning_rate)

    model.train(True)

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
