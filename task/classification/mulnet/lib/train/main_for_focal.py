import numpy as np
import os
import json

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms.transforms import CenterCrop

from .networks_for_focal import EmbeddingNet, SiameseNet
from .losses import ContrastiveLoss, NLL_OHEM, FocalLoss
from .trainer_for_focal import fit
from .datasets import CompDataset
from .datasets import SiameseComp


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
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # device = "cuda:{}".format(config.gpus)
    self_transform = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ]
    )
    train_dataset = CompDataset(
        config.train_file,
        train=True,
        config=config,
        transform=self_transform,
    )

    siamese_train_dataset = SiameseComp(
        train_dataset, config
    )  # Returns pairs of images and target same/different

    val_dataset = CompDataset(
        config.val_file,
        train=False,
        config=config,
        transform=val_transform,
    )
    siamese_val_dataset = SiameseComp(
        val_dataset, config
    )

    batch_size = config.batch_size
    kwargs = (
        {"num_workers": config.num_workers, "pin_memory": config.pin_memory}
        if use_cuda
        else {}
    )
    train_loader = torch.utils.data.DataLoader(
        siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        siamese_val_dataset, batch_size=1, shuffle=False
    )

    model = SiameseNet(config.cls_model_name, config.num_classes)
    if use_cuda:
        # print(torch.cuda.device_count())
        # print(torch.cuda.is_available())
        model.cuda()
        # model.to(device)

    focal_loss = FocalLoss(class_num=2)
    cls_fn = FocalLoss(class_num=config.num_classes)
    lr = config.learning_rate
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=config.steps, gamma=config.gamma
    )

    return train_loader, val_loader, model, focal_loss, cls_fn, optimizer, scheduler, self_transform


def train(config, sender, controller):
    print("begin train")
    use_cuda = config.use_cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    # 需要重写我们数据集的提取方法
    train_dataset = CompDataset(
        config.train_file,
        train=True,
        config=config,
        transform=transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    test_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ]
    )

    siamese_train_dataset = SiameseComp(
        train_dataset, config
    )  # Returns pairs of images and target same/different

    batch_size = config.batch_size
    kwargs = (
        {"num_workers": config.num_workers, "pin_memory": config.pin_memory}
        if use_cuda
        else {}
    )
    siamese_train_loader = torch.utils.data.DataLoader(
        siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )

    test_dataset = CompDataset(
        config.test_file,
        train=False,
        config=config,
        transform=test_transform,
    )
    siamese_test_dataset = SiameseComp(
        test_dataset, config
    )
    siamese_test_loader = torch.utils.data.DataLoader(
        siamese_test_dataset, batch_size=1, shuffle=False
    )

    siame_model = SiameseNet(config.cls_model_name, config.num_classes)
    if use_cuda:
        siame_model.cuda()
    focal_loss = FocalLoss(class_num=2)
    cls_fn = FocalLoss(class_num=config.num_classes)
    lr = config.learning_rate
    sgd_optimizer = optim.SGD(
        siame_model.parameters(),
        lr=lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    multilr_scheduler = lr_scheduler.MultiStepLR(
        sgd_optimizer, milestones=config.steps, gamma=config.gamma
    )

    fit(
        train_loader=siamese_train_loader,
        val_loader=siamese_test_loader,
        model=siame_model,
        num_classes=config.num_classes,
        loss_fn=focal_loss,
        cls_fn=cls_fn,
        optimizer=sgd_optimizer,
        scheduler=multilr_scheduler,
        n_epochs=config.epochs,
        cuda=use_cuda,
        log_interval=config.log_interval,
        metrics=[],
        start_epoch=0,
        cache_dir=config.cache_dir,
        model_bn_path=config.model_bn_path,
        train_status_sender=sender,
        train_controller=controller,
    )
    print("finish train")


if __name__ == "__main__":
    config = {}
    with open("config.json", "r") as f:
        config = json.load(f)
        f.close()
    train(config)
