import os
import time
import numpy as np
import torch
import cv2
import json
from typing import Dict,Any,List
from PIL import Image, ImageDraw, ImageFont

from riemann_interface.defaults import AbstractPredictor, AbstractObjectDetectPredictor

from .lib.detectors.detector_factory import detector_factory
from .lib.opts import opts
from flask import current_app

def createInstance(config_path):
    predictor = CenterNetPredictor()
    return predictor

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, pre_process_func):
    self.opt = opt
    self.img_dir = opt.test_img_dir
    self.data=[]
    for root,dirs,files in os.walk(self.img_dir) :
        for filename in files:
            self.data.append(os.path.join(root,filename))

    self.pre_process_func = pre_process_func
  
  def __getitem__(self, index):
    img_name = self.data[index]
    img_path = os.path.join(img_name)
    # print(img_path)
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in self.opt.test_scales:
    #   print("scale:",scale)
      images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_path, img_name, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.data)

class CenterNetPredictor(AbstractObjectDetectPredictor):

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
        
        if not self.setupPredictor(predictor_config):
            return False
        self.detector.load_model(os.path.join(model_dir_path,"model_best.pth"))
        self.model_path_=model_dir_path
        # current_app.logger.debug("loadmodel,current device info:{}".format(str(self.opt.device)))
        return True


    def predict(self, img: Any, info: Dict[str, Any] = None) -> Dict[str, Any]:
        # current_app.logger.debug("predict,current device info:{}".format(str(self.opt.device)))
        start_time = time.time()
        ret = self.detector.run(img)
        # print("ret",ret)
        result_dict=ret["results"]
        
        new_result_dict = {}
        # 先阈值过滤
        for key in result_dict:
            for bbox_score in result_dict[key]:
                bbox=bbox_score[0:4]
                score=bbox_score[4]
                if score <self.score_threshold[key-1]:
                    continue
                if key not in new_result_dict.keys():
                    new_result_dict[key] = []
                new_result_dict[key].append([bbox[0], bbox[1], bbox[2], bbox[3], score])
        
        # 再去除重复框
        results = self.filter_by_iou(new_result_dict)

        final_results={}

        #print("class names:",self.class_names_)
        for cat_id in results:
            cat_result=results[cat_id]
            filtered_cat_result=[]
            for bbox_score in cat_result:
                score=bbox_score[4]
                if score <self.score_threshold[cat_id-1] :
                    continue
                filtered_cat_result.append(bbox_score)
            final_results[self.class_names_[cat_id-1]]=filtered_cat_result
        ret["results"]=final_results
        # img = cv2.imread(img_path[0])
        # bbox = ret['results'][1][0][0:4]
        # img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        # cv2.imwrite(os.path.join('../results/', img_name[0]), img)
        end_time = time.time()
        # print('time:', (end_time - start_time))
        return ret
    def stackPredict(self, img_list: List[Any], info_list: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # current_app.logger.debug("current device info:{}".format(str(self.opt.device)))
        start_time = time.time()
        ret = self.detector.stack_run(img_list)
        result_dicts=ret["results"]
        final_results_list = []
        for result_dict in result_dicts:
            new_result_dict = {}
            # 先阈值过滤
            for key in result_dict:
                for bbox_score in result_dict[key]:
                    bbox=bbox_score[0:4]
                    score=bbox_score[4]
                    if score <self.score_threshold[key-1]:
                        continue
                    if key not in new_result_dict.keys():
                        new_result_dict[key] = []
                    new_result_dict[key].append([bbox[0], bbox[1], bbox[2], bbox[3], score])
            
            # 再去除重复框
            results = self.filter_by_iou(new_result_dict)

            final_results={}

            for cat_id in results:
                cat_result=results[cat_id]
                filtered_cat_result=[]
                for bbox_score in cat_result:
                    score=bbox_score[4]
                    if score <self.score_threshold[cat_id-1] :
                        continue
                    filtered_cat_result.append(bbox_score)
                final_results[self.class_names_[cat_id-1]]=filtered_cat_result
            final_results_list.append(final_results)
        ret["results"]=final_results_list
        end_time = time.time()
        # print('time:', (end_time - start_time))
        return ret
    
    def run_eval(self, img_dir, annotation_file, save_dir):
        import pycocotools.coco as coco
        from pycocotools.cocoeval import COCOeval
        coco_anno = coco.COCO(annotation_file)
        images = coco_anno.getImgIds()
        num_samples = len(images)
        class_name = ['__background__', ] + self.opt.class_names
        _valid_ids = [i for i in range(1, len(class_name)+1)]

        results = {}
        for index in range(num_samples):
            img_id = images[index]
            file_name = coco_anno.loadImgs(ids=[img_id])[0]['file_name']
            img_path = os.path.join(img_dir, file_name)
            ret = self.detector.run(img_path)
            results[img_id] = ret['results']

        def _to_float(x):
            return float("{:.2f}".format(x))

        def convert_eval_format(all_bboxes):
            # import pdb; pdb.set_trace()
            detections = []
            for image_id in all_bboxes:
                for cls_ind in all_bboxes[image_id]:
                    category_id = _valid_ids[cls_ind - 1]
                    for bbox in all_bboxes[image_id][cls_ind]:
                        bbox[2] -= bbox[0]
                        bbox[3] -= bbox[1]
                        score = bbox[4]
                        bbox_out  = list(map(_to_float, bbox[0:4]))

                        detection = {
                            "image_id": int(image_id),
                            "category_id": int(category_id),
                            "bbox": bbox_out,
                            "score": float("{:.2f}".format(score))
                        }
                        if len(bbox) > 5:
                            extreme_points = list(map(_to_float, bbox[5:13]))
                            detection["extreme_points"] = extreme_points
                        detections.append(detection)
            return detections

        def save_results(results, save_dir):
            json.dump(convert_eval_format(results), 
                        open('{}/results.json'.format(save_dir), 'w'))
        
        save_results(results, save_dir)
        coco_dets = coco_anno.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(coco_anno, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def testFolder(self,folder_path):
        forward_time = 0
        start_time = time.time()
        self.opt.test_img_dir=folder_path
        if not os.path.exists(os.path.join(folder_path+"-results")):
            os.makedirs(os.path.join(folder_path+"-results"))
        if hasattr(self,"data_loader"):
            del self.data_loader
        self.data_loader = torch.utils.data.DataLoader(
          PrefetchDataset(self.opt, self.detector.pre_process), 
          batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        if not hasattr(self,"color_list"):
            self.color_list=[]
            for i in range(256):
                self.color_list.append(tuple(((i*17)%256, (i+20)*67%256, (i+60)*101%256  )))
        for ind, (img_path, img_name, pre_processed_images) in enumerate(self.data_loader):
            t1 = time.time()
            ret = self.detector.run(pre_processed_images)
            t2 = time.time()
            forward_time += (t2 - t1)
            img = Image.open(img_path[0])
            
            result_dict=ret["results"]
            # print(result_dict)
            new_result_dict = {}
            # 先阈值过滤
            for key in result_dict:
                for bbox_score in result_dict[key]:
                    bbox=bbox_score[0:4]
                    score=bbox_score[4]
                    if score <self.score_threshold[key-1]:
                        continue
                    if key not in new_result_dict.keys():
                        new_result_dict[key] = []
                    new_result_dict[key].append([bbox[0], bbox[1], bbox[2], bbox[3], score])
            
            # 再去除重复框
            new_result_dict = self.filter_by_iou(new_result_dict)
            color_index=0
            for key in new_result_dict:
                for bbox_score in new_result_dict[key]:
                    bbox = bbox_score[0:4]
                    score = bbox_score[4]
                    # if score < self.score_threshold[key-1]: continue
                    draw=ImageDraw.Draw(img)
                    if not hasattr(self,"font"):
                        self.font = ImageFont.truetype("./ttf/Alibaba-PuHuiTi-Regular.ttf", 10, encoding="utf-8"  )
                    draw.text( (int(bbox[0]),int(bbox[1])), self.class_names_[key-1],fill=self.color_list[color_index],font=self.font)
                    draw.text( (int(bbox[0]),int(bbox[1])-20), str(score),fill=self.color_list[color_index],font=self.font)
                    
                    draw.rectangle(tuple(bbox),outline=self.color_list[color_index],width=4)
                color_index+=1
            out_img_path=folder_path+"-results/"+str( img_name[0]).split("/")[-1]
            # print("out imgname",out_img_path)
            img.save(out_img_path)
            
        end_time = time.time()
        print('avg time:', (end_time - start_time)/len(self.data_loader) ," s" )
        print('forward time:', forward_time / len(self.data_loader) ," s" )

    def modelPath(self):
        return self.model_path_

    def filter_by_iou(self, results):
        # results: 同一张图像中所有类别的boxes和scores: {cls_id: np.array([[x1, y1, x2, y2, score], [x1', y1', x2', y2', score'], ])}，经过阈值过滤后的预测结果
        def nms(dets, iou_thr):
            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            areas = (y2 -y1+1) * (x2 - x1+1)
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

                w = np.maximum(0, x22-x11+1)
                h = np.maximum(0, y22-y11+1)

                overlaps = w*h
                ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)

                idx = np.where(ious <= iou_thr)[0]
                index = index[(idx+1)]
            return keep
        
        # 如果两个bboxes的iou大于0.5，则保留置信度较大的那个bbox
        new_results = {}
        bboxes = [] # annotation_id: bbox
        cls_ids = {} # annotation_id: cls_id
        anno = 0
        if len(results) == 0: return results
        for cls_id, bbox in results.items():
            bboxes.extend(list(bbox))
            new_results[cls_id] = []
            for _ in range(len(bbox)):
                cls_ids[anno] = cls_id
                anno += 1
        bboxes = np.stack(bboxes)
        keep_idx = nms(bboxes, self.opt.iou_thr)
        for idx in keep_idx:
            new_results[cls_ids[idx]].append(bboxes[idx])
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

    def cal_threshold(self, results_file, annotation_file):
        # 计算置信度阈值, IoU阈值设为0.3
        import pycocotools.coco as coco
        from pycocotools.cocoeval import COCOeval
        coco_anno = coco.COCO(annotation_file)
        images_ids = coco_anno.getImgIds()
        dets = coco_anno.loadRes(results_file)
        cat_ids = coco_anno.getCatIds()
        default_conf_thr = {cid:0.2 for cid in cat_ids}

        def iou(dets, gts, iou_thr):
            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            areas = (y2 -y1+1) * (x2 - x1+1)
            scores = dets[:, 4]
            min_score = 1.5 # 随便取的，只要比1.0大就行
            for i in range(len(gts)):
                bbox = gts[i][0:4]
                area = (bbox[3]-bbox[1]+1) * (bbox[2]-bbox[0]+1)
            
                x11 = np.maximum(bbox[0], x1)
                y11 = np.maximum(bbox[1], y1)
                x22 = np.maximum(bbox[2], x2)
                y22 = np.maximum(bbox[3], y2)

                w = np.maximum(0, x22-x11+1)
                h = np.maximum(0, y22-y11+1)

                overlaps = w*h
                ious = overlaps / (area+areas - overlaps)

                max_iou = np.max(ious)
                max_iou_idx = np.argmax(ious)
                if max_iou >= iou_thr:
                    best_score = scores[max_iou_idx]
                    if best_score < min_score:
                        min_score = best_score
            return min_score
        
        for cid in cat_ids:
            min_s = 1.5
            for iid in images_ids:
                gt_ann_ids = coco_anno.getAnnIds([iid], [cid])
                if len(gt_ann_ids) == 0: continue
                det_ann_ids = dets.getAnnIds([iid], [cid])
                if len(det_ann_ids) == 0: continue
                gt_anns = coco_anno.loadAnns(gt_ann_ids)
                det_anns = dets.loadAnns(det_ann_ids)
                gt_bboxes = []
                det_bboxes = []
                for ann in gt_anns:
                    bbox = ann['bbox']
                    gt_bboxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], 1.0])
                for ann in det_anns:
                    bbox = ann['bbox']
                    score = ann['score']
                    det_bboxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], score])
                gt_bboxes = np.array(gt_bboxes)
                det_bboxes = np.array(det_bboxes)
                
                min_score = iou(det_bboxes, gt_bboxes, iou_thr=0.5)
                if min_score < min_s:
                    min_s = min_score
            if min_s < default_conf_thr[cid]:
                default_conf_thr[cid] = min_s
        #print(default_conf_thr)

    def save_results(self, img_id, results_dict, outputs):
        results = results_dict["results"]

        for cls_name in results.keys():
            for bbox_score in results[cls_name]:
                bbox = [int(bbox_score[0]), int(bbox_score[1]),
                        int(bbox_score[2] - bbox_score[0]), int(bbox_score[3] - bbox_score[1])]
                score = float(bbox_score[4])
                cls_id = self.class_names_.index(cls_name) + 1
                outputs.append({
                    "bbox": bbox,
                    "score": score,
                    "image_id": img_id,
                    "category_id": cls_id
                })