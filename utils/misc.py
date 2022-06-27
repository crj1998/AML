import random, csv
from datetime import datetime

import numpy as np

import torch
import torch.backends.cudnn as cudnn

def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cudnn.deterministic = True
        cudnn.benchmark = True
        print(torch.cuda.get_device_name(torch.cuda.current_device()))

def sample_cifar(dataset, num_per_class):
    targets = np.array(dataset.targets, dtype=int)
    C = np.max(targets) + 1
    assert targets.shape[0] % C == 0
    L = targets.shape[0] // C
    assert isinstance(num_per_class, int) and num_per_class <= L
    indices = np.argsort(targets)
    indices = indices.reshape(C, L)
    indices = indices[:, :num_per_class]
    indices = indices.flatten()
    dataset.data = dataset.data[indices]
    dataset.targets = targets[indices]
    return dataset

def subclass_cifar(dataset, selected):
    targets = np.array(dataset.targets, dtype=int)
    C = np.max(targets) + 1
    assert targets.shape[0] % C == 0
    flag = np.isin(targets, selected)
    dataset.data = dataset.data[flag]
    dataset.targets = targets[flag]
    for i, j in enumerate(selected):
        dataset.targets[dataset.targets==j] = i
    return dataset

class CSVwriter:
    def __init__(self, filename, head=None):
        self.filename = filename
        self.head = head
        if head is not None:
            self.register(head)
        
    def register(self, head):
        self.head = head

        with open(self.filename, mode="w", encoding="utf-8") as f:
            writer = csv.writer(f)
            if self.head and isinstance(head, (list, tuple)):
                writer.writerow(["Time", *self.head])
            else:
                pass
    
    def update(self, row):
        with open(self.filename, mode="a", encoding="utf-8") as f:
            writer = csv.writer(f)
            if isinstance(row, list):
                writer.writerow([datetime.today().strftime('%Y-%m-%d %H:%M:%S'), *row])
            elif isinstance(row, dict):
                writer.writerow([datetime.today().strftime('%Y-%m-%d %H:%M:%S'), *(row.get(k, "None") for k in self.head)])
            else:
                pass

