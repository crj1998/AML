"""
Adapted from https://github.com/csdongxian/AWP
"""

from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class AWP(nn.Module):
    def __init__(self, model, gamma, eps=1e-8, lr=0.01, weight_decay=1e-5):
        super().__init__()
        self.model = model
        self.proxy = deepcopy(model)
        self.proxy_optim = optim.SGD(self.proxy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = eps
        self.diff = OrderedDict()
    
    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def update_weight(self, coeff):
        for name, param in self.model.named_parameters():
            if name in self.diff.keys():
                param.add_(coeff * self.diff[name])

    def perturb(self, inputs, targets, weight=None):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        if weight is None:
            loss = - F.cross_entropy(self.proxy(inputs), targets, reduction="mean")
        else:
            loss = - (F.cross_entropy(self.proxy(inputs), targets, reduction="none") * weight).mean()
        
        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        self.diff = OrderedDict()
        for (old_k, old_w), (new_k, new_w) in zip(self.model.state_dict().items(), self.proxy.state_dict().items()):
            if len(old_w.size()) <= 1:
                continue
            if 'weight' in old_k:
                diff_w = new_w - old_w
                self.diff[old_k] =  diff_w / (diff_w.norm() + self.eps) * old_w.norm()

        self.update_weight( + 1.0 * self.gamma)
    
    def restore(self):
        self.update_weight( - 1.0 * self.gamma)

