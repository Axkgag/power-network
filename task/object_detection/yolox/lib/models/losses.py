#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
import torch
import torch.nn as nn
import torchvision


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        c_tl = torch.min(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        c_br = torch.max(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )
        cover = c_br - c_tl

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            area_c = torch.prod(cover, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "ciou":
            # inter_diag = torch.pow(target[:, 0] - pred[:, 0], 2) + torch.pow(target[:, 1] - pred[:, 1], 2)
            inter_diag = (target[:, 0] - pred[:, 0]) ** 2 + (target[:, 1] - pred[:, 1]) ** 2
            # outer_diag = torch.pow(cover[:, 0], 2) + torch.pow(cover[:, 1], 2)
            outer_diag = torch.clamp(cover[:, 0], min=0) ** 2 + torch.clamp(cover[:, 1], min=0) ** 2
            arctan = torch.atan(target[:, 2] / target[:, 3]) - torch.atan(pred[:, 2] / pred[:, 3])
            v = (4 / (math.pi ** 2)) * torch.pow(arctan, 2)
            with torch.no_grad():
                S = (iou > 0.5).float()
                alpha = S * v / (1 - iou + v)
            ciou = iou - inter_diag / outer_diag + alpha * v
            ciou = torch.clamp(ciou, min=-1.0, max=1.0)
            loss = 1 - ciou

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class EqlLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(EqlLoss, self).__init__()
        self.reduction = reduction
        self.becwith_log = nn.BCEWithLogitsLoss(reduction=self.reduction)
        self.freq = torch.tensor([0.006, 0.0017, 0.0316, 0.0413, 0.0548, 0.3989, 0.0089, 0.0242, 0.0432, 0.099, 0.1755, 0.0188, 0.0781, 0.0113])
        self.freq_lambda = 0.02

    def threshold_func(self, pred):
        n_i, n_c = pred.shape
        weight = pred.new_zeros(n_c)
        weight[self.freq < self.freq_lambda] = 1
        weight = weight.view(1, n_c).expand(n_i, n_c)
        return weight

    def forward(self, pred, target):
        eql_w = 1 - self.threshold_func(pred) * (1 - target)
        # cls_loss = self.becwith_log(pred, target)
        cls_loss = torchvision.ops.sigmoid_focal_loss(pred, target, reduction=self.reduction)
        return cls_loss * eql_w * 5.0