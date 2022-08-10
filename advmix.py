import os

import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.tensorboard import SummaryWriter

from model import builder as model_builder
from dataset import builder as dataset_builder
from optimizer import builder as optim_builder
from scheduler import builder as scheduler_builder
from attacker import builder as atk_builder

from utils.misc import setup_seed, CSVwriter
from utils.metric import LossMetric, AccuracyMetric
from utils.plot import plot_learning_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
 
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    #限制坐标区域不超过样本大小
 
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


@torch.enable_grad()
def train(epoch, model, dataloader, criterion, optimizer, **kwargs):
    global writer
    Loss = LossMetric()
    Acc = AccuracyMetric(10)
    ATK = atk_builder.build("PGD", steps=10, step_size=2/255, epsilon=8/255)
    beta = 1.0
    with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train ({epoch:2d})", ncols=120) as t:
        for i, (images, targets) in t:
            batch_size = targets.size(0)
            images, targets = images.to(device), targets.to(device)
            if len(targets.shape) == 1:
                labels = targets.clone()
                targets = F.one_hot(targets, num_classes=10)
            else:
                labels = targets.argmax(dim=-1)
            # non_tar_idx = torch.scatter(
            #     torch.ones((batch_size, 10), device=device), dim=-1, 
            #     index=labels.reshape(batch_size, 1), value=0.0
            # ).nonzero()[:, 1].reshape(batch_size, -1)
            
            # mask = torch.zeros_like(images)
            # shuffle = torch.randperm(batch_size, device=device)

            # lam = np.random.beta(beta, beta)
            # bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            # mask[:, :, bbx1:bbx2, bby1:bby2] = 1.0
            # lam = (bbx2 - bbx1) * (bby2 - bby1) / (images.size(2)*images.size(3))

            # d = round(32*np.random.uniform(0.2, 0.8))
            # s = np.random.randint(0, 32 - d)
            # if np.random.random()<0.5:
            #     mask[..., :, s:s+d] = 1.0
            # else:
            #     mask[..., s:s+d, :] = 1.0
            # lam = d/32

            # images  = mask * images  + (1 - mask) * images[shuffle]
            # targets = lam  * targets + (1 -  lam) * targets[shuffle]
            # difcls = (labels != labels[shuffle])
            delta = ATK(model, images, targets)

            model.train()
            logits = model(images + delta)
            loss = (- F.log_softmax(logits, dim=-1) * targets).sum(dim=-1).mean()
            # print(logits[difcls].shape, non_tar_idx[difcls].shape)
            # logits1 = torch.gather(logits[difcls], dim=-1, index=non_tar_idx[difcls])
            # logits2 = torch.gather(logits[difcls], dim=-1, index=non_tar_idx[shuffle][difcls])
            # labels1 = torch.where(labels[shuffle][difcls] < labels[difcls], labels[difcls]-1, labels[difcls])
            # labels2 = torch.where(labels[difcls] < labels[shuffle][difcls], labels[shuffle][difcls]-1, labels[shuffle][difcls])
            # loss = lam * criterion(logits1, labels1).mean() + (1.0 - lam) * criterion(logits2, labels2).mean()
            # loss = (- F.log_softmax(logits, dim=-1) * targets).sum(dim=-1).mean() + 0.3*(criterion(logits1, labels1).sum() + criterion(logits2, labels2).sum())/batch_size
            # loss = torch.topk((- F.log_softmax(logits, dim=-1) * targets).sum(dim=-1), 96).values.sum()/batch_size
            # logits_all = model(torch.cat([images + delta, images], dim=0))
            # logits, logits_ori = logits_all[:batch_size], logits_all[batch_size:]
            # loss = (- F.log_softmax(logits_ori, dim=-1) * targets).sum(dim=-1).mean() + beta * F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits_ori, dim=-1), reduction="batchmean")


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            Loss.update(loss.item(), batch_size)
            Acc.update(logits.detach().argmax(dim=-1).cpu().numpy(), targets.argmax(dim=-1).cpu().numpy())
            
            t.set_postfix({"Loss": f"{str(Loss)}", "Acc": f"{str(Acc)}"})

            acc = (logits.detach().argmax(dim=-1)==labels).sum().item()/batch_size

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
    transform = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomAffine(degrees=(0, +45)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    train_set = CIFAR10(root=args.data_path, train=True, download=False, transform=transform)
    test_set  = CIFAR10(root=args.data_path, train=False, download=False, transform=T.ToTensor())
    # , collate_fn = collate_fn
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    net = model_builder.build("resnet18", args.num_classes)
    net.load_state_dict(torch.load("results/resnet-18/PGDAT/best.pth"))
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
    $ CUDA_VISIBLE_DEVICES=0 python AML/advmix.py --epochs 150 --milestones 50 100 --suffix resnet-18/advmix --debug
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