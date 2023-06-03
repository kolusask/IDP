# https://github.com/bacnguyencong/rbm-pytorch/blob/master/rbm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import numpy as np


class RBM(nn.Module):
    r"""Restricted Boltzmann Machine.
    Args:
        n_vis (int, optional): The size of visible layer. Defaults to 784.
        n_hid (int, optional): The size of hidden layer. Defaults to 128.
        k (int, optional): The number of Gibbs sampling. Defaults to 1.
    """

    def __init__(self, n_vis=784, n_hid=128, k=1):
        """Create a RBM."""
        super(RBM, self).__init__()
        self.v = nn.Parameter(torch.randn(1, n_vis))
        self.h = nn.Parameter(torch.randn(1, n_hid))
        self.W = nn.Parameter(torch.randn(n_hid, n_vis))
        self.k = k
    
    def prop_forward(self, v):
        return torch.sigmoid(F.linear(v, self.W, self.h))
    
    def prop_backward(self, h):
        return torch.sigmoid(F.linear(h, self.W.t(), self.v))

    def free_energy(self, v):
        r"""Free energy function.
        .. math::
            \begin{align}
                F(x) &= -\log \sum_h \exp (-E(x, h)) \\
                &= -a^\top x - \sum_j \log (1 + \exp(W^{\top}_jx + b_j))\,.
            \end{align}
        Args:
            v (Tensor): The visible variable.
        Returns:
            FloatTensor: The free energy value.
        """
        v_term = torch.matmul(v, self.v.t())
        w_x_h = F.linear(v, self.W, self.h)
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return torch.mean(-h_term - v_term)

    def forward(self, v):
        r"""Compute the real and generated examples.
        Args:
            v (Tensor): The visible variable.
        Returns:
            (Tensor, Tensor): The real and generagted variables.
        """
        h = self.prop_forward(v)
        for _ in range(self.k):
            v_gibb = self.prop_backward(h)
            h = self.prop_forward(v_gibb)
        return v, v_gibb, h

def train_rbm(model: RBM, train_loader: data.DataLoader, n_epochs: int=20, lr: float=0.01, print_every=10):
    optimizer = torch.optim.Adam(model.parameters(), lr)

    model.train(True)

    for epoch in range(n_epochs):
        losses = []
        for data, _ in train_loader:
            v, v_gibbs, _ = model(data)
            loss = (model.free_energy(v) - model.free_energy(v_gibbs)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        
        if epoch % print_every == 0:
            print(f'Epoch {epoch}:\tloss={np.mean(losses)}')
    
    model.train(False)
    
    return model
