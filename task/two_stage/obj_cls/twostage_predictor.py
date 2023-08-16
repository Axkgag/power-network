import os
import time
import numpy
import cv2
import numpy as np
import json

import torch
import torch.backends.cudnn as cudnn
from riemann_interface.defaults import AbstractPredictor,AbstractObjectDetectPredictor

from typing import Dict, Any, List

def createInstance(predictor, classifier):
    predictor = Obj_ClsPredictor(predictor, classifier)
    return predictor

class Obj_ClsPredictor(AbstractPredictor):
    def __init__(self, predictor: AbstractObjectDetectPredictor, classifier: AbstractPredictor):
        self.predictor = predictor
        self.classifier = classifier

    def setupPredictor(self, config: Dict[str, Any]):
        return True

    def loadModel(self, model_dir_path: str) -> bool:
        model_path1 = os.path.join(model_dir_path, "stage1")
        model_path2 = os.path.join(model_dir_path, "stage2")

        if not (os.path.exists(model_path1) and os.path.exists(model_path2)):
            return False

        if not (self.predictor.loadModel(model_path1) and self.classifier.loadModel(model_path2)):
            return False

        return True

    def predict(self, img: Any, info: Dict[str, Any] = None) -> Dict[str, Any]:
        results = {}
        # needs = ["010202021", "010301031", "010401051", "010401061"]
        needs = ["010301031"]
        ignore_list = ["01040105", "01020202", "01040106", "01030103"]
        final_results = {"results": results}

        pred_outputs = self.predictor.smartPredict(img, info)
        pred_results = pred_outputs["results"]
        for pred_cls in pred_results.keys():
            for bbox_score in pred_results[pred_cls]:
                x0, y0, x1, y1, score = bbox_score
                if pred_cls in needs:
                    crop_img = img[int(y0):int(y1), int(x0):int(x1), :].copy()
                    if (crop_img.shape[0] == 0) or (crop_img.shape[1] == 0):
                        continue
                    cls_outputs = self.classifier.predict(crop_img, info)
                    cls_results = cls_outputs[0]["obj_class"]
                else:
                    cls_results = pred_cls
                if cls_results in ignore_list:
                    continue
                if cls_results not in results.keys():
                    results[cls_results] = []
                results[cls_results].append(bbox_score)

        final_results["results"] = results
        return final_results

    def save_results(self, img_id, results_dict, outputs):
        results = results_dict["results"]

        for cls_name in results.keys():
            for bbox_score in results[cls_name]:
                bbox = [int(bbox_score[0]), int(bbox_score[1]),
                        int(bbox_score[2] - bbox_score[0]), int(bbox_score[3] - bbox_score[1])]
                score = bbox_score[4]
                cls_id = self.cls_names.index(cls_name) + 1
                outputs.append({
                    "bbox": bbox,
                    "score": score,
                    "image_id": img_id,
                    "category_id": cls_id
                })
