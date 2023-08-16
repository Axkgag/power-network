import numpy as np
import os
import json

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torchvision import transforms

# from .trainer_for_focal import fit
from .datasets import CompDataset
from ..networks_for_focal import EfficientNet
from .utils import set_weight_decay


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {"epoch": epoch, "state_dict": state_dict}
    if not (optimizer is None):
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, path, _use_new_zipfile_serialization=False)


def setupForTrain(config):
    print("begin train")
    use_cuda = config.use_cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    train_transform = transforms.Compose(
        [
            # transforms.RandomResizedCrop((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.1)
        ]
    )

    val_transform = transforms.Compose(
        [
            # transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

    train_dataset = CompDataset(
        config.train_file,
        train=True,
        config=config,
        transform=train_transform,
    )

    val_dataset = CompDataset(
        config.train_file,
        train=False,
        config=config,
        transform=val_transform,
    )

    batch_size = config.batch_size
    kwargs = (
        {"num_workers": config.num_workers, "pin_memory": config.pin_memory}
        if use_cuda
        else {}
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )

    model = EfficientNet(config.arch, config.num_classes)
    if use_cuda:
        model.cuda()

    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion = nn.CrossEntropyLoss()

    parameters = set_weight_decay(
        model,
        0.00002,
        norm_weight_decay=0.0,
        custom_keys_weight_decay=None,
    )

    lr = config.learning_rate
    optimizer = optim.SGD(
        parameters,
        lr=lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs + 1, eta_min=0.0,
    )

    return train_loader, val_loader, model, criterion, optimizer, main_lr_scheduler, train_transform
