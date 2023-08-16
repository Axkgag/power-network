from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from unittest import result
import cv2
import json
import numpy as np
import time
import torch
from typing import Dict,Any,List

from .lib.opts import opts
from .lib.detectors.detector_factory import detector_factory
from riemann_interface.defaults import AbstractObjectDetectPredictor
import time
import math
import copy
# from flask import current_app

from .lib.utils.voc_eval_utils import voc_eval
from dotadevkit.polyiou import polyiou


def createInstance(config_path):
    predictor = RCenterNetPredictor()
    return predictor

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, pre_process_func):
    self.opt = opt
    self.img_dir = opt.test_img_dir
    self.data = np.loadtxt(opt.test_file, dtype=str)

    self.pre_process_func = pre_process_func
  
  def __getitem__(self, index):
    img_name = self.data[index]
    img_path = os.path.join(self.img_dir, img_name)
    print(img_path)
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in self.opt.test_scales:
      images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_path, img_name, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.data)

class RCenterNetPredictor(AbstractObjectDetectPredictor):

    def __init__(self, config: Dict[str, Any] = None):
        self.model_path_=""
        super().__init__(config)
        
    def setupPredictor(self, config: Dict[str, Any]):
        opt = opts()

        if config is None:
            return False
        self.config=config

        keys = ["arch", "class_names", "input_res", "score_threshold", "iou_thr",
                "num_classes", "fp16", "gpus", "mean", "std", "K"]
        for key in keys:
            if key not in config:
                print("{} is not in config".format(key))
                return False

        if not torch.cuda.is_available():
            config['gpus']='-1'
        
        self.opt = opts().update_dataset_info_and_set_heads(opt, config)
        self.opt.K = config["K"]
        
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['gpus']
        
        Detector = detector_factory[self.opt.task]
        self.class_names_=self.opt.class_names
        self.detector = Detector(self.opt)
        self.use_patch = self.setCropConfig(config)

        return True

    def loadModel(self, model_dir_path):
        predictor_config_path=os.path.join(model_dir_path,"predictor.json")
        weight_path=os.path.join(model_dir_path,"model_best.pth")
        if not os.path.exists(model_dir_path) or not os.path.exists(weight_path) or  not os.path.exists(predictor_config_path):
            return False
        with open(predictor_config_path,'r', encoding='utf-8') as f:
            predictor_config=json.load(f)
            f.close()

        if "score_threshold" in predictor_config:
            self.score_threshold=predictor_config["score_threshold"]
        
        # self.crop_size = predictor_config["crop_size"]
        # self.crop_stride = predictor_config["crop_stride"]
        if not self.setupPredictor(predictor_config):
            return False
        self.detector.load_model(os.path.join(model_dir_path,"model_best.pth"))
        self.model_path_=model_dir_path
        # current_app.logger.debug("loadmodel,current device info:{}".format(str(self.opt.device)))
        return True
    
    def predict(self, img: Any, info: Dict[str, Any] = None) -> Dict[str, Any]:
        ret = self.detector.run(img)
        result_dict=ret["results"]

        new_result_dict = {}
        # 先阈值过滤
        for key in result_dict:
            for bbox_score in result_dict[key]:
                bbox = bbox_score[0:4]
                ang = bbox_score[4]
                score = bbox_score[5]
                if score <self.score_threshold[key-1]:
                    continue
                if key not in new_result_dict.keys():
                    new_result_dict[key] = []
                new_result_dict[key].append([bbox[0], bbox[1], bbox[2], bbox[3], ang, score])
        
        results = self.filter_by_iou(new_result_dict)
        final_results={}

        for cat_id in results:
            cat_result=results[cat_id]
            filtered_cat_result=[]
            
            for bbox_score in cat_result:
                score = bbox_score[5]
                if score <self.score_threshold[cat_id-1] :
                    continue
                
                filtered_cat_result.append(bbox_score)
            final_results[self.class_names_[cat_id-1]]=filtered_cat_result

        ret["results"]=final_results 
        return ret

    def stackPredict(self, img_list: List[Any], info_list: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError
    
    def run_eval(self, img_dir, json_path, save_dir):
        import pycocotools.coco as coco
        coco_anno = coco.COCO(json_path)
        images_ids = coco_anno.getImgIds()
        file_names = [item["file_name"] for item in coco_anno.loadImgs(ids=images_ids)]
        res_save_dir = os.path.join(save_dir, "txt_results")
        if not os.path.exists(res_save_dir):
          os.mkdir(res_save_dir)
        
        gt_save_dir = os.path.join(save_dir, "txt_gt")
        if not os.path.exists(gt_save_dir):
          os.mkdir(gt_save_dir)

        self.dump_results(file_names, img_dir, res_save_dir)
        # self.load_results(json_dir, save_dir)

        with open(os.path.join(save_dir, "valset.txt"), "w") as f:
          for name in file_names:
            f.write(name.split(".")[0] + " " + "\n")
        
        converted_results = {}
        for file_name in file_names:
          file_name = file_name.split(".")[0]
          converted_results[file_name] = []

        for i, image_id in enumerate(images_ids):
          annIds = coco_anno.getAnnIds(imgIds=[image_id])
          anns = coco_anno.loadAnns(annIds)
          file_name = file_names[i].split(".")[0]
          for ann in anns:
            cat_id = self.class_names_[ann["category_id"]-1]
            x, y, w, h, ang = ann["bbox"]
            bbox_score = [x, y, x + w, y + h, ang] + [1]
            score, x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = self.convert_box(bbox_score)
            converted_results[file_name].append(str(x0n) + " " + str(y0n) + " " + str(x1n) + " " + str(y1n) 
                                  + " " + str(x2n) + " " + str(y2n) + " " + str(x3n) + " " + str(y3n) +  " " + cat_id + " " + "0" + "\n")

        for file_name in file_names:
          file_name = file_name.split(".")[0]
          with open(os.path.join(gt_save_dir, file_name+".txt"), 'w') as f:
            f.writelines(converted_results[file_name])

        detpath = res_save_dir + r'/{:s}.txt'
        annopath = gt_save_dir + r'/{:s}.txt'
        imagesetfile = save_dir + r'/valset.txt'
        classnames = self.class_names_
        classaps = []
        map = 0
        
        for classname in classnames:
            print('classname:', classname)
            rec, prec, ap = voc_eval(detpath,
                annopath,
                imagesetfile,
                classname,
                ovthresh=0.5,
                use_07_metric=True)
            map = map + ap
            print('ap: ', ap)
            classaps.append(ap)
        map = map/len(classnames)
        print('map:', map)
        classaps = 100*np.array(classaps)
        print('classaps: ', classaps)
      
    def dump_results(self, file_names, img_dir, save_dir):
        converted_results = {}
        for class_name in self.class_names_:
          converted_results[class_name] = []
        
        for file_name in file_names:
          img = cv2.imread(os.path.join(img_dir, file_name))  # 文件夹名需修改

          ret = self.smartPredict(img) # 需选择patchPredict或predict模式
          # ret = self.predict(img)

          det_results = ret["results"]
          # save into txt format
          for cat_id in det_results:
            cat_result = det_results[cat_id]
            for bbox_score in cat_result:
                score, x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = self.convert_box(bbox_score)
                converted_results[cat_id].append(file_name.split(".")[0] + " " + str(score) + " " + str(x0n) + " " + str(y0n) + " " + str(x1n) + " " + str(y1n) 
                                  + " " + str(x2n) + " " + str(y2n) + " " + str(x3n) + " " + str(y3n) + "\n")
            
        for class_name in self.class_names_:
          with open(os.path.join(save_dir, class_name+".txt"), 'w') as f:
            f.writelines(converted_results[class_name])

    def load_results(self, json_dir, save_dir):
        converted_results = {}
        for class_name in self.class_names_:
            converted_results[class_name] = []

        with open(os.path.join(json_dir, "c_outputs.json"), "r") as f:
            c_res = json.load(f)

        for res in c_res:
            bbox = res["bbox"]
            pre_score = res["score"]
            cat_id = str(res["category_id"])
            img_name = res["image_id"]
            bbox_score = bbox + [pre_score]
            score, x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = self.convert_box(bbox_score)
            converted_results[cat_id].append(
                str(img_name) + " " + str(score) + " " + str(x0n) + " " + str(y0n) + " " + str(
                    x1n) + " " + str(y1n)
                + " " + str(x2n) + " " + str(y2n) + " " + str(x3n) + " " + str(y3n) + "\n")

        for class_name in self.class_names_:
            with open(os.path.join(save_dir, class_name + ".txt"), 'w') as f:
                f.writelines(converted_results[class_name])

    def modelPath(self):
        return self.model_path_
    
    def convert_box(self, bbox_score):
        bbox = bbox_score[0:4]
        ang = bbox_score[4]
        score = bbox_score[5]
        result = np.array([int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2),int(bbox[2]-bbox[0]),int(bbox[3]-bbox[1]), ang])
        x, y, width, height = int(result[0]), int(result[1]), int(result[2]), int(result[3])
        anglePi = result[4]/180 * math.pi
        anglePi = anglePi if anglePi <= math.pi else anglePi - math.pi

        cosA = math.cos(anglePi)
        sinA = math.sin(anglePi)
        
        x1=x-0.5*width   
        y1=y-0.5*height
        
        x0=x+0.5*width 
        y0=y1
        
        x2=x1            
        y2=y+0.5*height 
        
        x3=x0   
        y3=y2
        
        x0n= int((x0 -x)*cosA -(y0 - y)*sinA + x)
        y0n = int((x0-x)*sinA + (y0 - y)*cosA + y)
        
        x1n= int((x1 -x)*cosA -(y1 - y)*sinA + x)
        y1n = int((x1-x)*sinA + (y1 - y)*cosA + y)
        
        x2n= int((x2 -x)*cosA -(y2 - y)*sinA + x)
        y2n = int((x2-x)*sinA + (y2 - y)*cosA + y)
        
        x3n= int((x3 -x)*cosA -(y3 - y)*sinA + x)
        y3n = int((x3-x)*sinA + (y3 - y)*cosA + y)
        return score, x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n

    def save_img(self, img, results, filename, output_dir):
        from PIL import Image, ImageDraw, ImageFont
        from matplotlib import pyplot as plt
        image = Image.fromarray(img)
        draw = ImageDraw.Draw(image)
        for cat_id in results:
            cat_result = results[cat_id]
            for bbox_score in cat_result:
                score, x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = self.convert_box(bbox_score)
                draw.line([(x0n, y0n),(x1n, y1n)], fill=(0, 0, 255), width=3) # blue  横线
                draw.line([(x1n, y1n),(x2n, y2n)], fill=(255, 0, 0), width=3) # red    竖线
                draw.line([(x2n, y2n),(x3n, y3n)], fill= (0, 0, 255), width=3)
                draw.line([(x0n, y0n), (x3n, y3n)], fill=(255, 0, 0), width=3)

                ttf_path = "./ttf/Alibaba-PuHuiTi-Light.ttf"
                ttf = ImageFont.truetype(ttf_path, 25)
                text = cat_id + ": " + str(score)
                draw.text((x0n, y0n), text, font=ttf, fill=(255, 255, 0))
        image.save(os.path.join(output_dir, os.path.split(filename)[-1]))

    def show_bboxes(self, img, results):
        from PIL import Image, ImageDraw
        from matplotlib import pyplot as plt
        image = Image.fromarray(img)
        draw = ImageDraw.Draw(image)
        for cat_id in results:
          cat_result = results[cat_id]
          for bbox_score in cat_result:
              score, x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = self.convert_box(bbox_score)
              draw.line([(x0n, y0n),(x1n, y1n)], fill=(0, 0, 255), width=3)  # blue  横线
              draw.line([(x1n, y1n),(x2n, y2n)], fill=(255, 0, 0), width=3)  # red    竖线
              draw.line([(x2n, y2n),(x3n, y3n)], fill= (0, 0, 255), width=3)
              draw.line([(x0n, y0n), (x3n, y3n)], fill=(255, 0, 0), width=3)
        
        plt.imshow(image)
        plt.show()

    def filter_by_iou(self, results):
        def nms(dets, iou_thr):
          bb = dets[:, 1:]
          scores = dets[:, 0]
          keep = []
          areas = np.abs((bb[:, 2] - bb[:, 0])*(bb[:, 5] - bb[:, 3]) - (bb[:, 3] - bb[:, 1])*(bb[:, 4] - bb[:, 2]))

          areas_scores = np.array([(area, score) for area, score in zip(areas, scores)],
                                  dtype=[("area", float), ("score", float)])
          index = areas_scores.argsort(order=("area", "score"))[::-1]

          bb_xmin = np.min(bb[:, 0::2], axis=1)
          bb_ymin = np.min(bb[:, 1::2], axis=1)
          bb_xmax = np.max(bb[:, 0::2], axis=1)
          bb_ymax = np.max(bb[:, 1::2], axis=1)

          rect_areas = (bb_ymax -bb_ymin+1) * (bb_xmax - bb_xmin+1)

          while index.size > 0:
            i = index[0]
            keep.append(i)
            
            x11 = np.maximum(bb_xmin[i], bb_xmin[index[1:]])
            y11 = np.maximum(bb_ymin[i], bb_ymin[index[1:]])
            x22 = np.minimum(bb_xmax[i], bb_xmax[index[1:]])
            y22 = np.minimum(bb_ymax[i], bb_ymax[index[1:]])

            w = np.maximum(0, x22-x11+1)
            h = np.maximum(0, y22-y11+1)

            overlaps = w*h
            ious = overlaps / (rect_areas[i] + rect_areas[index[1:]] - overlaps)
            ious_bak = copy.deepcopy(ious)
            for idx, iou in enumerate(ious_bak):
              if iou > 0:
                ious[idx] = polyiou.iou_poly(polyiou.VectorDouble(bb[index[1:]][idx]), polyiou.VectorDouble(bb[i]))
            
            idx = np.where(ious <= iou_thr)[0]
            index = index[(idx+1)]
          return keep

        new_results = {}
        bboxes = []
        bboxes_scores = []
        cls_ids = {}
        anno = 0
        if len(results) == 0: return results
        for cls_id, bbox_scores in results.items():
          bboxes_scores.extend(list(bbox_scores))
          new_results[cls_id] = []
          for idx, bbox_score in enumerate(bbox_scores):
            bboxes.append(list(self.convert_box(bbox_score)))
            cls_ids[anno] = cls_id
            anno += 1

        bboxes = np.stack(bboxes)
        keep_idx = nms(bboxes, self.opt.iou_thr)
        for idx in keep_idx:
          new_results[cls_ids[idx]].append(bboxes_scores[idx])
        return new_results

    def filter_by_iou_plus(self, results):
        # results: 同一张图像中所有类别的boxes和scores:
        # {cls_id: np.array([[x1, y1, x2, y2, score], [x1', y1', x2', y2', score'], ])}，
        # 经过阈值过滤后的预测结果
        def nms_plus(dets, iou_thr):
            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            areas = (y2 - y1 + 1) * (x2 - x1 + 1)
            scores = dets[:, 4]
            keep = []
            areas_scores = np.array([(area, score) for area, score in zip(areas, scores)],
                                    dtype=[("area", float), ("score", float)])
            index = areas_scores.argsort(order=("area", "score"))[::-1]
            while index.size > 0:
                i = index[0]
                keep.append(i)
                x11 = np.maximum(x1[i], x1[index[1:]])
                y11 = np.maximum(y1[i], y1[index[1:]])
                x22 = np.minimum(x2[i], x2[index[1:]])
                y22 = np.minimum(y2[i], y2[index[1:]])

                w = np.maximum(0, x22 - x11 + 1)
                h = np.maximum(0, y22 - y11 + 1)

                overlaps = w * h
                ious = overlaps / np.minimum(areas[i], areas[index[1:]])

                idx = np.where(ious <= iou_thr)[0]
                index = index[(idx + 1)]
            return keep

        # 如果两个bboxes的iou大于0.5，则保留置信度较大的那个bbox
        new_results = {}
        bboxes = []  # annotation_id: bbox
        cls_ids = {}  # annotation_id: cls_id
        anno = 0
        if len(results) == 0:
            return results
        for cls_id, bbox in results.items():
            bboxes.extend(list(bbox))
            new_results[cls_id] = []
            for _ in range(len(bbox)):
                cls_ids[anno] = cls_id
                anno += 1
        bboxes = np.stack(bboxes)
        keep_idx = nms_plus(bboxes, self.opt.iou_thr)
        for idx in keep_idx:
            new_results[cls_ids[idx]].append(bboxes[idx])
        return new_results
      
    def verify_filter_by_iou_function(self, n_center_points=3, disturbance=30):
        import random
        w, h = 512, 512
        img = np.ones((w, h, 3), dtype=np.uint8) * 255
        base_w, base_h = 100, 100
        results = {}
        for cat_id in self.class_names_:
          results[cat_id] = []
          for _ in range(n_center_points):
            cx = np.random.choice(list(range(25, w-25)))
            cy = np.random.choice(list(range(25, h-25)))
            for i in range(5):
              ang = np.random.choice(range(0, 180, 1))
              score = random.random()
              x = cx + np.random.choice(range(-disturbance, disturbance, 1))
              y = cy + np.random.choice(range(-disturbance, disturbance, 1))
              bb_w = base_w + np.random.choice(range(-disturbance, disturbance, 1))
              bb_h = base_h + np.random.choice(range(-disturbance, disturbance, 1))
              results[cat_id].append([x - bb_w/2., y - bb_h/2., x + bb_w/2., y + bb_h/2., ang, score])
        
        self.show_bboxes(img, results)
        new_results = self.filter_by_iou(results)
        self.show_bboxes(img, new_results)

    def save_results(self, img_id, results_dict, outputs):
        results = results_dict["results"]

        for cls_name in results.keys():
            for bbox_score in results[cls_name]:
                bbox = [int(bbox_score[0]), int(bbox_score[1]),
                        int(bbox_score[2] - bbox_score[0]), int(bbox_score[3] - bbox_score[1]),
                        int(bbox_score[4])]
                score = float(bbox_score[5])
                cls_id = self.class_names_.index(cls_name) + 1
                outputs.append({
                    "bbox": bbox,
                    "score": score,
                    "image_id": img_id,
                    "category_id": cls_id
                })

    # def patchPredict(self, img: Any, info: Dict[str, Any] = None):
    #
    #     h, w, c = img.shape
    #     h_r, w_r = 3, 3  # 滑动次数
    #     stride = 256  # 滑动步长
    #     image_resize = cv2.resize(img, dsize=(512 * w_r - stride * (w_r - 1), 512 * h_r - stride * (h_r - 1)))
    #
    #     new_results = {}
    #     for i in range(h_r):
    #         for j in range(w_r):
    #             img_crop = image_resize[i * 512 - i * stride: (i + 1) * 512 - i * stride,
    #                        j * 512 - j * stride: (j + 1) * 512 - j * stride]
    #             # 一个滑窗预测的结果
    #             ret_crop = self.predict(img_crop)
    #             ret = ret_crop["results"]
    #             if len(ret) == 0: continue
    #             for cls_id, bbox_scores in ret.items():
    #                 for n in range(len(bbox_scores)):
    #                     bbox = bbox_scores[n][0:4]
    #                     ang = bbox_scores[n][4]
    #                     score = bbox_scores[n][5]
    #                     x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    #                     x1 = max(0, np.floor(x1).astype('int32')) + np.floor(512 * j - j * stride).astype('int32')
    #                     y1 = max(0, np.floor(y1).astype('int32')) + np.floor(512 * i - i * stride).astype('int32')
    #                     x2 = min(img_crop.shape[1], np.floor(x2).astype('int32')) + np.floor(512 * j - j * stride).astype(
    #                         'int32')
    #                     y2 = min(img_crop.shape[0], np.floor(y2).astype('int32')) + np.floor(512 * i - i * stride).astype(
    #                         'int32')
    #                     h_scale = h / (h_r * 512 - (h_r - 1) * stride)
    #                     w_scale = w / (w_r * 512 - (w_r - 1) * stride)
    #                     x1 = np.floor(x1 * w_scale).astype('int32')
    #                     y1 = np.floor(y1 * h_scale).astype('int32')
    #                     x2 = np.floor(x2 * w_scale).astype('int32')
    #                     y2 = np.floor(y2 * h_scale).astype('int32')
    #                     bbox_score = [x1, y1, x2, y2, ang, score]
    #                     if cls_id not in new_results.keys():
    #                         new_results[cls_id] = []
    #                     new_results[cls_id].append(bbox_score)
    #     results = self.filter_by_iou(new_results)
    #
    #     final_results = {}
    #     for cat_id in results:
    #         cat_result = results[cat_id]
    #         filtered_cat_result = []
    #
    #         for bbox_score in cat_result:
    #             score = bbox_score[5]
    #             if score < 0.3:
    #                 continue
    #
    #             filtered_cat_result.append(bbox_score)
    #         final_results[cat_id] = filtered_cat_result
    #     return final_results
    #

