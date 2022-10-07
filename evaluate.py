import os, argparse

from tqdm import tqdm
from functools import partial

import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder

from attacker import builder as atk_builder
from attacker import AutoAttack as aa

from model import builder as model_builder
from utils.misc import setup_seed

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", type=str, default="resnet-18", choices=["resnet-18", "preactresnet-18", "wideresnet-28-10"])
parser.add_argument("-w", "--weight", type=str, required=True)
parser.add_argument("-d", "--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100", "SVHN", "Tiny-ImageNet"])
parser.add_argument("-a", "--attacker", choices=["CLEAN", "FGSM", "IFGSM", "PGD", "AA"], default="CLEAN", type=str)
parser.add_argument("-f", "--folder", type=str, required=True)
parser.add_argument("--eps", default=8, type=int)
parser.add_argument("--num_classes", default=10, type=int)
parser.add_argument("--cuda", action="store_true", default=False)
# parser.add_argument("--normalize", action="store_true", default=False)
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()

batch_size = 128
args.eps = args.eps/255

args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
setup_seed(args.seed)


net = model_builder.build(args.model, args.num_classes)
net.load_state_dict(torch.load(args.weight))
net = net.to(device)
net = net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == "CIFAR10":
    dataset = CIFAR10(root=args.folder, train=False, download=False, transform=transform)
elif args.dataset == "CIFAR100":
    dataset = CIFAR100(root=args.folder, train=False, download=False, transform=transform)
elif args.dataset == "SVHN":
    dataset = SVHN(root=args.folder, split="test", download=False, transform=transform)
elif args.dataset == "Tiny-ImageNet":
    dataset = ImageFolder(root=args.folder, transform=transform)
else:
    raise ValueError("Unknown dataset!")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# if args.normalize:
#     normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
# else:
#     normalize = None


@torch.enable_grad()
def pgd(model, images, labels, steps, step_size, epsilon, restarts):
    model.eval()
    max_delta = torch.zeros_like(images)
    robust = torch.ones_like(labels, dtype=torch.bool)
    imgs = images[robust]
    for _ in range(restarts):
        if _ > 0:
            delta = torch.zeros_like(imgs).uniform_(-epsilon, epsilon)
            delta = torch.clamp(imgs + delta, 0.0, 1.0) - imgs
        else:
            delta = torch.zeros_like(imgs)

        for _ in range(steps):
            delta.requires_grad_(True)
            loss = F.cross_entropy(model(imgs + delta), labels[robust])
            grad = torch.autograd.grad(loss, [delta])[0].detach()

            delta = delta.detach() + step_size * grad.sign()
            delta = torch.clamp(delta, - epsilon, + epsilon)
            delta = torch.clamp(imgs + delta, 0.0, 1.0) - imgs

        with torch.no_grad():
            non_robust = (model(imgs + delta).argmax(dim=-1) != labels[robust])
            non_robust_index = robust.nonzero(as_tuple=False)[non_robust].squeeze()
            max_delta[non_robust_index] = delta[non_robust]
            robust[non_robust_index] = False
        if robust.sum().item() == 0: 
            break
        imgs = images[robust]

    return max_delta

def PGD_20_10(model, images, labels):
    return pgd(model, images, labels, steps=20, step_size=args.eps/8, epsilon=args.eps, restarts=10)
def PGD10(model, images, labels):
    return pgd(model, images, labels, steps=10, step_size=args.eps/4, epsilon=args.eps, restarts=1)
def FGSM(model, images, labels):
    return pgd(model, images, labels, steps=1, step_size=args.eps, epsilon=args.eps, restarts=1)

PGD_20_10 = partial(pgd, steps=20, step_size=args.eps/8, epsilon=args.eps, restarts=10)
PGD10 = partial(pgd, steps=10, step_size=args.eps/4, epsilon=args.eps, restarts=1)
FGSM = partial(pgd, steps=1, step_size=args.eps, epsilon=args.eps, restarts=1)
adversary = aa(net, norm="Linf", eps=args.eps, version="standard", verbose=False)
adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
# adversary.apgd.n_restarts = 1
# adversary.apgd_targeted.n_restarts = 1
# adversary.apgd_targeted.n_target_classes = 1
# adversary.fab.n_restarts = 2
aa = adversary.run_standard_evaluation

total = correct = 0
with torch.no_grad():
    with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Eval {args.attacker}") as t:
        for i, (images, labels) in t:
            images, labels = images.to(device), labels.to(device)
            if args.attacker == "AA":
                images = aa(images, labels)
            elif args.attacker == "PGD":
                images = images + PGD_20_10(net, images, labels)
            elif args.attacker == "IFGSM":
                images = images + PGD10(net, images, labels)
            elif args.attacker == "FGSM":
                images = images + FGSM(net, images, labels)
            else:
                pass
            preds = net(images).detach().argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            t.set_postfix(Acc=f"{correct/total:4.2%}")
print(f"Robust under {args.attacker}: {correct/total:4.2%}")


"""
python AML/evaluate.py --cuda -m resnet-18 -w 'results/resnet-18/Flooding_12_long@10-06 12:29/last.pth' -d CIFAR10 -f ../data -a AA
"""