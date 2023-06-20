import os
import gc
import argparse
import torch
import torchvision
import wandb

from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from data.mcdominoes import get_mcdominoes, MCDOMINOES
from data.celeba import get_celeba, CELEBA
from data.logger import log_data, LOGGER


if __name__ == '__main__':
    # --- Parser ---
    parser = argparse.ArgumentParser()

    # ---Training Args ---
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', help='weight decay', type=float, default=1e-3)
    parser.add_argument("--batch_size", help='Batch Size', type=int, default=16)

    # --- Model Args ---
    parser.add_argument('--architecture', help='resnet50 or resnet18', type=str, default='resnet50')
    parser.add_argument("--pretrained", action='store_true', help="Use pretrained model")

    # --- Data Args ---
    parser.add_argument('--data_dir', help='data path', type=str, default='/hpctmp/e0200920')
    parser.add_argument("--dataset", type=str, default="mcdominoes",
                        help="Which dataset to use: [celeba]")
    parser.add_argument('--spurious_strength', help='Strength of spurious correlation', type=float, default=0.95)
    parser.add_argument('--seed', help='Seed', type=int, default=0)

    args = parser.parse_args()
    print(args, flush=True)

    # --- Random Seed ---
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Data ---
    data_dir = os.path.join(args.data_dir, args.dataset)
    # X should be treated as a list, Y and P are np arrays
    if args.dataset == "mcdominoes":
        n_epoch = 100
        nClasses = 10
        dim = (64, 32, 3)
        handler = MCDOMINOES
        X_tr, Y_tr, P_tr, X_val, Y_val, P_val, X_te, Y_te, P_te = get_mcdominoes(args.spurious_strength,
                                                                                 data_dir, args.seed)
    else:
        n_epoch = 50
        nClasses = 2
        dim = (224, 224, 3)
        handler = CELEBA
        X_tr, Y_tr, P_tr, X_val, Y_val, P_val, X_te, Y_te, P_te = get_celeba(args.spurious_strength,
                                                                             data_dir, args.seed)

    log_data(X_tr, Y_tr, P_tr, handler, args.dataset, "Train Dataset")
    log_data(X_val, Y_val, P_val, handler, args.dataset, "Validation Dataset")
    log_data(X_te, Y_te, P_te, handler, args.dataset, "Test Dataset")

    loader_tr = DataLoader(handler(X_tr, torch.Tensor(Y_tr).long(), torch.Tensor(P_tr).long(), isTrain=True),
                           shuffle=True, batch_size=args.batch_size)
    loader_val = DataLoader(handler(X_val, torch.Tensor(Y_val).long(), torch.Tensor(P_val).long(), isTrain=False),
                            shuffle=False, batch_size=args.batch_size)

    # --- Model ---
    if args.architecture == "resnet18":
        net = torchvision.models.resnet18(pretrained=args.pretrained)
    else:
        net = torchvision.models.resnet50(pretrained=args.pretrained)
    net.fc = torch.nn.Linear(net.fc.in_features, nClasses)
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # --- Metrics ---
    logger = LOGGER(["Train Accuracy", "Train Minority Accuracy", "Train Majority Accuracy", "Train Loss",
                     "Train Minority Loss", "Train Majority Loss", "Val Accuracy", "Val Minority Accuracy",
                     "Val Majority Accuracy"])

    # --- Training Loop ---
    for epoch in range(n_epoch):
        train_acc, train_minority_acc, train_majority_acc, train_loss, train_minority_loss, train_majority_loss \
            = 0, 0, 0, 0, 0, 0
        # --- Train ---
        net.train()
        for batch_idx, (x, y, p, idxs) in enumerate(loader_tr):
            x, y, p = Variable(x.cuda()), Variable(y.cuda()), Variable(p.cuda())
            optimizer.zero_grad()
            out = net(x)
            loss = F.cross_entropy(out, y, reduction=None)
            average_loss = torch.mean(loss)
            average_loss.backward()
            optimizer.step()
            # --- Update Metrics
            train_acc += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()
            train_minority_acc += torch.sum((torch.max(out[y != p], 1)[1] == y[y != p]).float()).data.item()
            train_majority_acc += torch.sum((torch.max(out[y == p], 1)[1] == y[y == p]).float()).data.item()
            train_loss += torch.sum(loss).item()
            train_minority_loss += torch.sum(loss[y != p]).item()
            train_majority_loss += torch.sum(loss[y == p]).item()

        logger.log({"Train Accuracy": train_acc/len(loader_tr),
                   "Train Minority Accuracy": train_minority_acc/len(loader_tr),
                   "Train Majority Accuracy": train_majority_acc/len(loader_tr),
                   "Train Loss": train_loss/len(loader_tr),
                   "Train Minority Loss": train_minority_loss/len(loader_tr),
                   "Train Majority Loss": train_majority_loss/len(loader_tr)})

        # --- Evaluate ---
        val_acc, val_minority_acc, val_majority_acc = 0, 0, 0
        net.eval()
        for batch_idx, (x, y, p, idxs) in enumerate(loader_val):
            x, y, p = Variable(x.cuda()), Variable(y.cuda()), Variable(p.cuda())
            out = net(x)
            # --- Update Metrics
            val_acc += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()
            val_minority_acc += torch.sum((torch.max(out[y != p], 1)[1] == y[y != p]).float()).data.item()
            val_majority_acc += torch.sum((torch.max(out[y == p], 1)[1] == y[y == p]).float()).data.item()
        logger.log({"Val Accuracy": val_acc/len(loader_tr),
                   "Val Minority Accuracy": val_minority_acc/len(loader_tr),
                   "Val Majority Accuracy": val_majority_acc/len(loader_tr)})
