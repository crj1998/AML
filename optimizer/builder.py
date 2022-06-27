import torch.optim as optim


optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam
}


def build(name, model, **optim_kwargs):
    assert name in optimizers, f"Unknown optimizer: {name}."

    no_decay = optim_kwargs.pop("no_decay", [])
    weight_decay = optim_kwargs.pop("weight_decay", 0.0)
    grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = optimizers[name](grouped_parameters, **optim_kwargs)

    return optimizer