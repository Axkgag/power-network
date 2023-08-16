#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
from torch import nn
import numpy as np

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck

class SpatialNorm(nn.Module):

    def __init__(self, p=0.5, num_groups=64, eps=1e-5):
        super(SpatialNorm, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0
        self.T = 5
        self.num_groups = num_groups

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        if len(x.shape) == 2:
            t = t.repeat(x.shape[0], 1)
        else:
            t = t.repeat(x.shape[0], 1, 1)
        return t

    def matrix_power3(self, Input):
        B = torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def groupwhitening(self, input):

        size = input.size()
        num_groups = min(self.num_groups, size[1])
        # channel shuffle
        idx = torch.randperm(size[1], dtype=torch.int64)
        reverse_idx = torch.zeros_like(idx, dtype=torch.int64)\
            .scatter_(0, idx, torch.arange(0, size[1], dtype=torch.int64))
        input_shu = input[:, idx, :, :]

        x = input_shu.view(size[0], num_groups, size[1] // num_groups, *size[2:])

        x = x.view(size[0], num_groups, -1)  # 64 x 64 x -1
        IG, d, m = x.size()
        mean = x.mean(-1, keepdim=True)
        x_mean = x - mean
        P = [torch.Tensor([]) for _ in range(self.T + 1)]
        sigma = x_mean.matmul(x_mean.transpose(1, 2)) / m  # 协方差

        P[0] = torch.eye(d).to(x).expand(sigma.shape)
        M_zero = sigma.clone().fill_(0)
        trace_inv = torch.addcmul(M_zero, sigma, P[0]).sum((1, 2), keepdim=True).reciprocal_()  # 矩阵迹的倒数
        sigma_N = torch.addcmul(M_zero, sigma, trace_inv)
        for k in range(self.T):
            P[k + 1] = torch.baddbmm(P[k], self.matrix_power3(P[k]), sigma_N, beta=1.5, alpha=-0.5)
            # (input, batch1, batch2, beta, alpha): out = beta * input + alpha * (batch1 @ batch2)
        wm = torch.addcmul(M_zero, P[self.T], trace_inv.sqrt())
        # print(wm.shape) # 64 x 64 x 64
        wm_inv = self.b_inv(wm)

        sqrtvar_wm_inv = self.sqrtvar(wm_inv)
        wm_inv_reparam = self._reparameterize(wm_inv, sqrtvar_wm_inv)
        a1 = torch.triu(wm_inv_reparam)
        a2 = torch.triu(wm_inv_reparam, diagonal=1)
        gamma = a1 + a2.transpose(1, 2)

        sqrtvar_mean = self.sqrtvar(mean)
        beta = self._reparameterize(mean, sqrtvar_mean)
        y = wm.matmul(x_mean)
        y = gamma.matmul(y) + beta

        output1 = y.view(size[0], num_groups, size[1] // num_groups, *size[2:])
        output1 = output1.view_as(input)
        output = output1[:, reverse_idx, :, :]

        return output

    def b_inv(self, b_mat):
        eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
        # b_inv, _ = torch.solve(eye, b_mat)
        b_inv = torch.linalg.solve(eye, b_mat)
        return b_inv

    def forward(self, x):
        if (not self.training) or np.random.random() > self.p:
            return x
        x = self.groupwhitening(x)
        return x

class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
        uncertainty=0.0
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

        self.attn1 = SpatialNorm(p=uncertainty)
        self.attn2 = SpatialNorm(p=uncertainty)
        self.attn3 = SpatialNorm(p=uncertainty)
        self.attn4 = SpatialNorm(p=uncertainty)
        self.attn5 = SpatialNorm(p=uncertainty)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        x = self.attn1(x)
        outputs["stem"] = x

        x = self.dark2(x)
        x = self.attn2(x)
        outputs["dark2"] = x

        x = self.dark3(x)
        x = self.attn3(x)
        outputs["dark3"] = x

        x = self.dark4(x)
        x = self.attn4(x)
        outputs["dark4"] = x

        x = self.dark5(x)
        x = self.attn5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
