import os
import random
from PIL import Image

from tqdm import tqdm
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100

from model import builder as model_builder
from dataset import builder as dataset_builder
from optimizer import builder as optim_builder
from scheduler import builder as scheduler_builder
from attacker import builder as atk_builder

from dataset.randaug import RandAugment


from utils.misc import setup_seed, CSVwriter
from utils.metric import LossMetric, AccuracyMetric
from utils.plot import plot_learning_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getitem(self, index):
    img, target = self.data[index], self.targets[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)

    if self.transform is not None:
        if isinstance(self.transform, list):
            img = [transform(img) for transform in self.transform]
        else:
            img = [self.transform(img)]

    if self.target_transform is not None:
        target = self.target_transform(target)
    return *img, target

CIFAR10.__getitem__ = getitem




@torch.enable_grad()
def train(epoch, model, dataloader, criterion, optimizer, **kwargs):
    global writer

    Loss = LossMetric()
    Acc = AccuracyMetric(10)
    PGD10 = atk_builder.build("PGD", steps=10, step_size=2/255, epsilon=8/255)
    with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train ({epoch:2d})", ncols=120) as t:
        for i, (images, labels) in t:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            delta = PGD10(model, images, labels)

            model.train()
            logits = model(images + delta)
            loss = criterion(logits, labels).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            Loss.update(loss.item(), labels.size(0))
            Acc.update(logits.detach().argmax(dim=-1).cpu().numpy(), labels.cpu().numpy())
            
            t.set_postfix({"Loss": f"{str(Loss)}", "Acc": f"{str(Acc)}"})

            acc = (logits.detach().argmax(dim=-1)==labels).sum().item()/labels.size(0)

            iters = len(dataloader)*(epoch-1) + i
            writer.add_scalar("Train/Loss", loss.item(), iters)
            writer.add_scalar("Train/Acc", acc, iters)

    return Loss.item(), Acc.item()

@torch.no_grad()
def test(model, dataloader, attack=None):
    Acc = AccuracyMetric(10)
    model.eval()
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        if callable(attack):
            delta = attack(model, images, labels)
        else:
            delta = torch.zeros_like(images)
        
        logits = model(images+delta)
        Acc.update(logits.detach().argmax(dim=-1).cpu().numpy(), labels.cpu().numpy())

    return Acc.item()

def main(args):
    setup_seed(42)
    global epoch

    def patch_randaug(batch):
        vmin, vmax = 0.0, epoch/150
        images, images_aug, labels = zip(*batch)  # transposed
        
        images = torch.stack(images, dim=0)
        images_aug = torch.stack(images_aug, dim=0)
        labels = torch.LongTensor(labels)
        N, C, H, W = images.shape
        masks = torch.zeros_like(images)
        
        for i in range(N):
            w, h = round(W*np.random.uniform(vmin, vmax)), round(H*np.random.uniform(vmin, vmax))
            x, y = np.random.randint(0, W - w), np.random.randint(0, H - h)
            masks[i, :, x:x+w, y:y+h] = 1.0
        images = (1 - masks) * images + masks * images_aug
        return images, labels

    transform = [
        T.Compose([
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]),
        T.Compose([
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            RandAugment(4, 10),
            T.ToTensor(),
        ])
    ]

    train_set = CIFAR10(root=args.data_path, train=True, download=False, transform=transform)
    test_set  = CIFAR10(root=args.data_path, train=False, download=False, transform=T.ToTensor())

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=patch_randaug)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)


    net = model_builder.build("resnet18", args.num_classes)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim_builder.build("SGD", net, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = scheduler_builder.build("stepwise", optimizer, milestones=args.milestones, gamma=0.1, last_epoch=-1)

    
    Train, Clean, Pgd10 = [], [], []
    best_pgd10 = 0

    global writer
    writer = SummaryWriter(args.out)
    recorder = CSVwriter(f"{args.out}/record.csv")
    recorder.register(["Epoch", "LR", "Loss", "Acc", "Clean", "PGD10"])
    
    PGD10 = atk_builder.build("PGD", steps=10, step_size=2/255, epsilon=8/255)
    for epoch in range(1, args.epochs+1):
        loss, acc = train(epoch, net, train_loader, criterion, optimizer)
        clean = test(net, test_loader, None)
        pgd10 = test(net, test_loader, PGD10)

        scheduler.step()

        Train.append(acc)
        Clean.append(clean)
        Pgd10.append(pgd10)

        lr = round(scheduler.get_last_lr()[0], 3)
        writer.add_scalar("Test/clean", clean, epoch)
        writer.add_scalar("Test/pgd10", pgd10, epoch)
        writer.add_scalar("LR", lr, epoch)
        recorder.update([epoch, lr, round(loss, 4), round(acc, 4), round(clean, 4), round(pgd10, 4)])
        plot_learning_curve(args.out, args.epochs, TRAIN=Train, CLEAN=Clean, PGD10=Pgd10)
        torch.save(net.state_dict(), f"{args.out}/last.pth")
        if pgd10 > best_pgd10:
            best_pgd10 = pgd10
            torch.save(net.state_dict(), f"{args.out}/best.pth")

    writer.close()

if __name__ == "__main__":
    """
    $ CUDA_VISIBLE_DEVICES=0 python AML/DynAug.py --epochs 150 --milestones 50 100 --suffix rn18/PGDATpathaug --debug
    $ CUDA_VISIBLE_DEVICES=0 python AML/PGDAT.py --epochs 150 --milestones 50 100 --suffix wideresnet-34-16/PGDAT
    """
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("--suffix", required=True, type=str, help="File suffix")
    parser.add_argument('--data_path', default="../data", type=str, metavar='PATH', help='data path')
    parser.add_argument('--num_classes', metavar='INT', default=10, type=int, help="number of classes")
    parser.add_argument('--batch_size', metavar='INT', default=128, type=int, help="batch size")
    parser.add_argument('--epochs', metavar='INT', default=100, type=int, help="number of epoch")
    parser.add_argument('--lr', default=0.1, type=float, metavar='FLOAT', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='FLOAT', help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='FLOAT', help='weight decay (default: 1e-4)')
    parser.add_argument("--milestones", default=[50, 75], type=int, nargs='+', metavar='LIST')
    parser.add_argument("--num_per_class", default=5000, type=int, metavar='INT')
    parser.add_argument("--debug", action="store_true")
        
    args = parser.parse_args()
    
    args.out = os.path.join("results", args.suffix if args.debug else f"{args.suffix}@{datetime.today().strftime('%m-%d %H:%M')}")
    os.makedirs(args.out, exist_ok=True)
    main(args)