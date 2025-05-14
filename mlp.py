import argparse

import torch
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import numpy as np
import sys

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)
    


def flag(model_forward, perturb_shape, y, args, optimizer, device, criterion) :
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-2e-3, 2e-3).to(device)
    perturb.requires_grad_()
    out = forward(perturb)
    loss = criterion(out, y)
    loss /= 3
    
    loss.backward()  

    
    for _ in range(2):
        
        perturb_data = perturb.detach() + 2e-3 * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out = forward(perturb)
        loss = criterion(out, y)
        loss /= 3
        loss.backward()  


    optimizer.step()

    return loss, perturb.detach()


def _mlp(model, x, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x[train_idx])
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

def train_flag(model, x, data, optimizer, args, device) :
    train_idx = data.train_mask.to(device)
    
    
    forward = lambda perturb : model(x[train_idx] + perturb) 

    model_forward = (model, forward)
    y = data.y[data.train_mask].squeeze()
    
    loss, perturb = flag(model_forward, x[train_idx].shape, y, args, optimizer, device, F.nll_loss)

    return loss.item(), perturb

@torch.no_grad()
def test_mlp(model, x, data, y_true, split_idx, evaluator):
    model.eval()

    out = model(x)
    pred = out.argmax(dim=-1, keepdim=True)
    accs = []
    for _, mask in data("train_mask", "val_mask", "test_mask"):
        
        if evaluator:
            acc = evaluator.eval({"y_true": data.y[mask], "y_pred": pred.unsqueeze(1)})["acc"]

        else:
            acc = pred.eq(data.y[mask].squeeze()).sum().item() / mask.sum().item()
        accs.append(acc)

    
    return accs

