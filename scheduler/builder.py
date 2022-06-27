
import math

from torch.optim.lr_scheduler import LambdaLR, MultiStepLR

def cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=7. / 16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


schedulers = {
    "cosine": cosine_schedule_with_warmup,
    "stepwise": MultiStepLR
}



def build(name, optimizer, **scheduler_kwargs):
    assert name in schedulers, f"Unknown scheduler: {name}."
    return schedulers[name](optimizer, **scheduler_kwargs)
