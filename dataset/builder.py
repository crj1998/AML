import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100

from dataset.randaug import RandAugment


# def getitem(self, index):
#     img, target = self.data[index], self.targets[index]

#     # doing this so that it is consistent with all other datasets
#     # to return a PIL Image
#     img = Image.fromarray(img)

#     if self.transform is not None:
#         if isinstance(self.transform, list):
#             img = [transform(img) for transform in self.transform]
#         else:
#             img = [self.transform(img)]

#     if self.target_transform is not None:
#         target = self.target_transform(target)
#     return *img, target

# CIFAR10.__getitem__ = getitem

# def patch_randaug(batch):
#     vmin, vmax = 0.05, 0.95
#     images, images_aug, labels = zip(*batch)  # transposed
    
#     images = torch.stack(images, dim=0)
#     images_aug = torch.stack(images_aug, dim=0)
#     labels = torch.LongTensor(labels)
#     N, C, H, W = images.shape
#     masks = torch.zeros_like(images)
    
#     for i in range(N):
#         w, h = round(W*np.random.uniform(vmin, vmax)), round(H*np.random.uniform(vmin, vmax))
#         x, y = np.random.randint(0, W - w), np.random.randint(0, H - h)
#         masks[i, :, x:x+w, y:y+h] = 1.0
#     images = (1 - masks) * images + masks * images_aug
#     return images, labels

def build(name, data_path, batch_size, num_workers, split_valid=0):
    name = name.lower()
    if name == "cifar10":

        # transform = [
        #     T.Compose([
        #         T.RandomCrop(32, padding=4, padding_mode="reflect"),
        #         T.RandomHorizontalFlip(),
        #         T.ToTensor(),
        #     ]),
        #     T.Compose([
        #         T.RandomCrop(32, padding=4, padding_mode="reflect"),
        #         T.RandomHorizontalFlip(),
        #         RandAugment(4, 10),
        #         T.ToTensor(),
        #     ])
        # ]
        transform = T.Compose([
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        train_set = CIFAR10(root=data_path, train=True, download=False, transform=transform)
        valid_set = CIFAR10(root=data_path, train=False, download=False, transform=T.ToTensor())
        test_set  = CIFAR10(root=data_path, train=False, download=False, transform=T.ToTensor())

        # targets = np.array(train_set.targets)
        # flag = np.zeros(targets.shape[0], dtype=bool)
        # flag[targets.argsort().reshape(10, -1)[:, -split_valid:].reshape(-1)] = True
        # train_set.data, train_set.targets = train_set.data[~flag], targets[~flag]

        # targets = np.array(valid_set.targets)
        # flag = np.zeros(targets.shape[0], dtype=bool)
        # flag[targets.argsort().reshape(10, -1)[:, -split_valid:].reshape(-1)] = True
        # valid_set.data, valid_set.targets = valid_set.data[flag], targets[flag]
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader


