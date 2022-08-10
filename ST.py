import os

from tqdm import tqdm
from datetime import datetime

import wandb



import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

from model import builder as model_builder
from dataset import builder as dataset_builder
from optimizer import builder as optim_builder
from scheduler import builder as scheduler_builder

from utils.misc import setup_seed, CSVwriter
from utils.metric import LossMetric, AccuracyMetric
from utils.plot import plot_learning_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.enable_grad()
def train(epoch, model, dataloader, criterion, optimizer, **kwargs):
    # global writer

    Loss = LossMetric()
    Acc = AccuracyMetric(10)
    
    with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train ({epoch:2d})", ncols=120, disable=not kwargs.get("pbar", True)) as t:
        for i, (images, labels) in t:
            batch_size = labels.size(0)
            images, labels = images.to(device), labels.to(device)

            model.train()
            logits = model(images)
            loss = criterion(logits, labels).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            Loss.update(loss.item(), batch_size)
            Acc.update(logits.detach().argmax(dim=-1).cpu().numpy(), labels.cpu().numpy())
            
            t.set_postfix({"Loss": f"{str(Loss)}", "Acc": f"{str(Acc)}"})

            # acc = (logits.detach().argmax(dim=-1)==labels).sum().item()/batch_size
            # iters = len(dataloader)*(epoch-1) + i
            # writer.add_scalar("Train/Loss", loss.item(), iters)
            # writer.add_scalar("Train/Acc", acc, iters)

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
    wandb.init(
        project = "AML", entity = "maze", 
        name = f"ST_{args.model}_{args.dataset}",
        config = {
            k: getattr(args, k) for k in ['model', 'dataset', 'num_classes', 'batch_size', 'optimizer', 'epochs', 'lr', 'momentum', 'weight_decay', 'scheduler', 'milestones', 'suffix', 'seed']
        }
    )

    net = model_builder.build(args.model, args.num_classes)
    net = net.to(device)

    train_loader, valid_loader, test_loader = dataset_builder.build(args.dataset, args.data_path, args.batch_size, num_workers=4)

    criterion = nn.CrossEntropyLoss(reduction="none")

    optimizer = optim_builder.build(args.optimizer, net, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = scheduler_builder.build(args.scheduler, optimizer, milestones=args.milestones, gamma=0.1, last_epoch=-1)

    
    Test_acc, Train_acc = [], []
    bets_acc = 0

    # global writer
    # writer = SummaryWriter(args.out)
    recorder = CSVwriter(f"{args.out}/record.csv")
    recorder.register(["Epoch", "LR", "Loss", "Acc", "Test"])
    
    for epoch in range(1, args.epochs+1):
        lr = round(scheduler.get_last_lr()[0], 3)
        loss, train_acc = train(epoch, net, train_loader, criterion, optimizer, pbar=args.pbar)
        test_acc = test(net, test_loader, None)

        scheduler.step()

        Test_acc.append(test_acc)
        Train_acc.append(train_acc)

        
        # writer.add_scalar("Test/acc", test_acc, epoch)
        # writer.add_scalar("LR", lr, epoch)
        wandb.log({"Train/Loss": loss}, step=epoch)
        wandb.log({"Train/Acc": train_acc}, step=epoch)
        wandb.log({"Test/Acc": test_acc}, step=epoch)
        wandb.log({"LR": lr}, step=epoch)
        if not args.pbar:
            print(f"Epoch {epoch:3d}: lr={lr:.3f} loss={loss:.3f} train={train_acc:.2%} test={test_acc:.2%}")
        recorder.update([epoch, round(lr, 3), round(loss, 3), round(train_acc, 4), round(test_acc, 4)])
        plot_learning_curve(args.out, args.epochs, TRAIN=Train_acc, TEST=Test_acc)
        torch.save(net.state_dict(), f"{args.out}/last.pth")
        if test_acc > bets_acc:
            bets_acc = test_acc
            torch.save(net.state_dict(), f"{args.out}/best.pth")

    # writer.close()

if __name__ == "__main__":
    """
    $ CUDA_VISIBLE_DEVICES=0 python AML/ST.py --epochs 100 --milestones 50 75 --suffix rn18/ST
    """
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("--suffix", required=True, type=str, help="File suffix")
    parser.add_argument("--model", default="resnet-18", type=str, help="model")
    parser.add_argument("--dataset", default="cifar10", type=str, help="dataset")
    parser.add_argument('--data_path', default="../data", type=str, metavar='PATH', help='data path')
    parser.add_argument('--num_classes', metavar='INT', default=10, type=int, help="number of classes")
    parser.add_argument('--batch_size', metavar='INT', default=128, type=int, help="batch size")
    parser.add_argument("--optimizer", default="SGD", type=str, help="optimizer")
    parser.add_argument("--scheduler", default="stepwise", type=str, help="scheduler")
    parser.add_argument('--epochs', metavar='INT', default=100, type=int, help="number of epoch")
    parser.add_argument('--lr', default=0.1, type=float, metavar='FLOAT', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='FLOAT', help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='FLOAT', help='weight decay (default: 1e-4)')
    parser.add_argument("--milestones", default=[50, 75], type=int, nargs='+', metavar='LIST')
    parser.add_argument('--seed', metavar='INT', default=42, type=int, help="random seed")
    parser.add_argument('--pbar', default=False, action="store_true", help="progress bar")
    # parser.add_argument("--num_per_class", default=5000, type=int, metavar='INT')

    args = parser.parse_args()

    # print(dir(args))
    args.debug = True if "dev" in args.suffix else False
    args.out = os.path.join("aml_result", args.suffix if args.debug else f"{args.suffix}@{datetime.today().strftime('%m-%d %H:%M')}")
    os.makedirs(args.out, exist_ok=True)
    main(args)