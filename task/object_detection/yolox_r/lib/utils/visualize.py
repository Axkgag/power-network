#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np
import math
import torch
from ..utils.boxes import convert_box

__all__ = ["vis"]


def vis(img, results, conf=0.5, class_names=None):
    for cls_name in results:
        for bbox_score in results[cls_name]:
            score, x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = convert_box(bbox_score)
            if score < conf:
                continue

            # text = cls_id
            text = cls_name + ": " + str(score)
            txt_color = (0, 0, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            # xc, yc = int((bbox_score[0] + bbox_score[2]) / 2), int((bbox_score[1] + bbox_score[3]) / 2)
            cv2.putText(img, text, (x0n, y0n + txt_size[1]), font, 1, txt_color, thickness=2)

            cv2.line(img, (int(x0n), int(y0n)), (int(x1n), int(y1n)), (0, 0, 255), 2, 4)
            cv2.line(img, (int(x1n), int(y1n)), (int(x2n), int(y2n)), (255, 0, 0), 2, 4)
            cv2.line(img, (int(x2n), int(y2n)), (int(x3n), int(y3n)), (0, 0, 255), 2, 4)
            cv2.line(img, (int(x0n), int(y0n)), (int(x3n), int(y3n)), (255, 0, 0), 2, 4)
    return img
