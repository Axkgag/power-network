import torch
import torch.nn.functional as F
import numpy as np
import os

def train_epoch(train_loader, model, criterion, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (img2, label) in enumerate(train_loader):
        if cuda:
            img2 = img2.cuda()
            label = label.cuda()

        optimizer.zero_grad()
        scores = model(img2)

        if label is not None:
            loss_clses = criterion(scores, label)
        
        loss_cls = loss_clses[0] if type(loss_clses) in (tuple, list) else loss_clses
        loss = loss_cls
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(img2), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def val(train_loader, model, criterion, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.eval()
    losses = []
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (img2, label, _, _) in enumerate(train_loader):
            if cuda:
                img2 = img2.cuda()
                label = label.cuda()

            scores = model(img2)

            if label is not None:
                loss_clses = criterion(scores, label)

            loss = loss_clses[0] if type(loss_clses) in (tuple, list) else loss_clses
            losses.append(loss.item())
            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(img2), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), np.mean(losses))
                for metric in metrics:
                    message += '\t{}: {}'.format(metric.name(), metric.value())

                print(message)
                losses = []

    total_loss /= (batch_idx + 1)
    return total_loss

def test(val_loader, model, num_classes, cuda):
    with torch.no_grad():
        model.eval()
        sum_ = 0
        for batch_idx, (img2, label, _, _) in enumerate(val_loader):
            print(str(batch_idx) + '\r', end="")
            if cuda:
                img2 = img2.cuda()
            scores = model(img2)[0]
            pred_cls = torch.argmax(scores)
            sum_ += int(pred_cls == label)
        return sum_ / len(val_loader)
