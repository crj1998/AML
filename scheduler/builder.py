
import math

from torch.optim.lr_scheduler import LambdaLR, MultiStepLR

def cosine_schedule_with_warmup(optimizer, warmup, total, lr_min, last_epoch=-1):
    lr_lambda = lambda i: lr_min + (i / warmup if i < warmup else (1-lr_min) * (1.0 + math.cos((i-warmup)/(total-warmup)*math.pi)) / 2) 
    return LambdaLR(optimizer, lr_lambda, last_epoch)


schedulers = {
    "cosine": cosine_schedule_with_warmup,
    "stepwise": MultiStepLR
}


def build(name, optimizer, **scheduler_kwargs):
    assert name in schedulers, f"Unknown scheduler: {name}."
    return schedulers[name](optimizer, **scheduler_kwargs)
