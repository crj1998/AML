"""
Adapted from https://github.com/imrahulr/adversarial_robustness_pytorch

https://github.com/imrahulr/adversarial_robustness_pytorch/blob/main/gowal21uncovering/utils/watrain.py
"""


from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_bn_momentum(model, momentum=1):
    """ Set the value of momentum for all BN layers.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.momentum = momentum

class SWA(nn.Module):
    def __init__(self, model, momentum):
        super().__init__()
        self.model = model
        self.proxy = deepcopy(model)
        self.momentum = momentum

    def forward(self, x):
        return self.proxy(x)

    @torch.no_grad()
    def update_weight(self, steps, warmup_steps=0, dynamic_decay=True):
        factor = int(steps >= warmup_steps)
        if dynamic_decay:
            delta = steps - warmup_steps
            decay = min(self.momentum, (1. + delta) / (10. + delta)) if 10. + delta != 0 else self.momentum
        else:
            decay = self.momentum

        decay *= factor
        for model, proxy in zip(self.model.parameters(), self.proxy.parameters()):
            # model.data = proxy.data * decay + proxy.data * (1.0 - decay)
            model.data.copy_(decay * model.data + (1. - decay) * proxy.data)

    @torch.no_grad()
    def update_bn(self):
        """ Update batch normalization layers.
        """
        self.eval()
        for module1, module2 in zip(self.model.modules(), self.proxy.modules()):
            if isinstance(module1, nn.modules.batchnorm._BatchNorm):
                module1.running_mean = module2.running_mean.clone()
                module1.running_var = module2.running_var.clone()
                module1.num_batches_tracked = module2.num_batches_tracked