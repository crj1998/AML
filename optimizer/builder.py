import torch.optim as optim
import torch.nn as nn

optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW
}

def separate_parameters(model):
    weight_decay = set()
    no_weight_decay = set()

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters():
            full_param_name = f"{module_name}.{param_name}" if module_name else param_name
            if isinstance(module, nn.Linear) and ("weight" in param_name):
                weight_decay.add(full_param_name)

    for name, param in model.named_parameters():
        if name not in weight_decay:
            no_weight_decay.add(name)
    # sanity check
    assert len(weight_decay & no_weight_decay) == 0
    assert len(weight_decay) + len(no_weight_decay) == len(list(model.parameters()))

    return no_weight_decay

def build(name, model, **optim_kwargs):
    assert name in optimizers, f"Unknown optimizer: {name}."

    no_decay = optim_kwargs.pop("no_decay", [])
    weight_decay = optim_kwargs.pop("weight_decay", 0.0)
    grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if n not in no_decay], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if n in no_decay], "weight_decay": 0.0}
    ]
    optimizer = optimizers[name](grouped_parameters, **optim_kwargs)

    return optimizer