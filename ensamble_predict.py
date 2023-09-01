import argparse
import os
import time
from typing import Any, Dict
import cv2
import numpy as np
import json
import logging

from riemann_interface.defaults import AbstractObjectDetectPredictor
from colors import _COLORS
from copy import deepcopy
from pycocotools.coco import COCO
from task.object_detection.yolox.yolox_predictor import YoloxPredictor
from utils.riemann_coco_eval import getAPForRes

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir',
                    default='/data/gauss/lyh/datasets/power_networks/test/model/06-08/',
                    help='test_dir, dir of dataset be tested')
parser.add_argument('-t', '--test_dir',
                    default='/data/gauss/lyh/datasets/power_networks/test/',
                    help='test_dir, dir of dataset be tested')
parser.add_argument('-s', '--save_result',
                    default=False,
                    action='store_true',
                    help='save result or not')

class EnsamblePredictor(AbstractObjectDetectPredictor):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.predictor_list = []
        self.crop_w = 0
        self.crop_h = 0
        self.stride_x = 0
        self.stride_y = 0

    def loadModel(self, model_dir):
        if not os.path.exists(model_dir):
            logging.error("model dir is not exists!")
            return False
        
        folder_names = os.listdir(model_dir)
        for folder in folder_names:
            predictor_dir = os.path.join(model_dir, folder)
            predictor = YoloxPredictor()
            if not predictor.loadModel(predictor_dir):
                logging.error("load single predictor model fail!")
                return False
            else:
                self.predictor_list.append(predictor)
        
        return True
    
    def filter_by_iou_plus(self, results):
        def nms_iou(detections, iou_thr):
            x1, y1 = detections[:, 0], detections[:, 1]
            x2, y2 = detections[:, 2], detections[:, 3]
            scores = detections[:, 4]
            areas = (y2 - y1 + 1) * (x2 - x1 + 1)
            nms_index = []
            areas_scores = np.array([(area, score) for area, score in zip(areas, scores)],
                                    dtype=[("area", float), ("score", float)])
            index = areas_scores.argsort(order=("area", "score"))[::-1]

            while index.size > 0:
                i = index[0]
                nms_index.append(i)
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
            return nms_index

        new_results = {}
        bboxes = []
        cls_ids = {}
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
        keep_idx = nms_iou(bboxes, 0.4)
        for idx in keep_idx:
            new_results[cls_ids[idx]].append(bboxes[idx])
        return new_results

    def predict(self, img):
        h, w, c = img.shape

        if self.crop_w and self.crop_h and self.stride_x and self.stride_y:
            crop_x_list = [self.crop_w]
            crop_y_list = [self.crop_h]
            stride_x_list = [self.stride_x]
            stride_y_list = [self.stride_y]
        else:
            crop_x_list = [w, 800]
            crop_y_list = [h, 800]
            stride_x_list = [0.8 * x for x in crop_x_list]
            stride_y_list = [0.8 * y for y in crop_y_list]

        new_results = {}  # 所有bbox的预测结果

        for crop_x, crop_y, stride_x, stride_y in zip(crop_x_list, crop_y_list, stride_x_list, stride_y_list):
            crop_x = int(crop_x)
            crop_y = int(crop_y)
            stride_x = int(stride_x)
            stride_y = int(stride_y)

            w_r, h_r = int((w - crop_x) / stride_x + 1), int((h - crop_y) / stride_x + 1)
            W, H = crop_x + stride_x * (w_r - 1), crop_y + stride_y * (h_r - 1)
            image_resize = cv2.resize(img, dsize=(W, H))

            # 汇总box
            for i in range(h_r):
                for j in range(w_r):
                    img_crop = image_resize[i * stride_y: i * stride_y + crop_y,
                               j * stride_x: j * stride_x + crop_x]
                    # 一个滑窗预测的结果
                    # ret_crop = self.predict(img_crop)

                    for predictor in self.predictor_list:
                        ret_crop = predictor.predict(img_crop)

                        ret = ret_crop["results"]
                        if len(ret) == 0:
                            continue
                        # cls_id, bbox_scores分别是预测类别和预测结果
                        # cls_id.size=1, bbox_scores.size=5(包含x1,y1,x2,y2,angle,probability)
                        for cls_id, bbox_scores in ret.items():
                            for n in range(len(bbox_scores)):
                                bbox = bbox_scores[n][0:4]
                                if len(bbox_scores[n]) == 6:  # if mode == 'Rotation_Detection':
                                    ang = bbox_scores[n][4]
                                    score = bbox_scores[n][5]
                                else:
                                    score = bbox_scores[n][4]
                                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                                # 映射到切割前的图片上（1024*1024）
                                x1 = max(0, np.floor(x1).astype('int32')) + np.floor(j * stride_x).astype('int32')
                                y1 = max(0, np.floor(y1).astype('int32')) + np.floor(i * stride_y).astype('int32')
                                x2 = min(img_crop.shape[1], np.floor(x2).astype('int32')) + np.floor(
                                    j * stride_x).astype(
                                    'int32')
                                y2 = min(img_crop.shape[0], np.floor(y2).astype('int32')) + np.floor(
                                    i * stride_y).astype(
                                    'int32')
                                h_scale = h / H
                                w_scale = w / W
                                # 映射到resize前的图像上（原始尺寸）
                                x1 = np.floor(x1 * w_scale).astype('int32')
                                y1 = np.floor(y1 * h_scale).astype('int32')
                                x2 = np.floor(x2 * w_scale).astype('int32')
                                y2 = np.floor(y2 * h_scale).astype('int32')
                                if len(bbox_scores[n]) == 6:
                                    bbox_score = [x1, y1, x2, y2, ang, score]
                                else:
                                    bbox_score = [x1, y1, x2, y2, score]

                                if cls_id not in new_results.keys():
                                    new_results[cls_id] = []
                                new_results[cls_id].append(bbox_score)

        # 进行nms过滤
        results = self.filter_by_iou_plus(new_results)

        # # 过滤掉置信度小于0.3的框
        # final_results = {}
        # for cat_id in results:
        #     cat_result = results[cat_id]
        #     filtered_cat_result = []

        #     for bbox_score in cat_result:
        #         if len(bbox_score) == 6:
        #             score = bbox_score[5]
        #         else:
        #             score = bbox_score[4]
        #         # if score < 0.2:
        #         #     continue

        #         filtered_cat_result.append(bbox_score)
        #     final_results[cat_id] = filtered_cat_result

        ret = {"results": results}
        return ret


CLASS_NAME = [
        "010101021",
        "010101031",
        "010101061",
        "010101071",
        "010201041",
        "010202021",
        "010301021",
        "010301011",
        "010301031",
        "010401041",
        "010401051",
        "010401061",
        "050101041",
        "060101051"
    ]

def visual(img_path, img, save_dir, results_dict, current_time):
    results = results_dict["results"]
    result_image = vis(img, results, 0.5, CLASS_NAME)

    save_folder = os.path.join(
        save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    save_file_name = os.path.join(save_folder, os.path.basename(img_path))
    cv2.imencode('.jpg', result_image)[1].tofile(save_file_name)

def save_results(img_id, results_dict, outputs):
    results = results_dict["results"]

    for cls_name in results.keys():
        for bbox_score in results[cls_name]:
            bbox = [int(bbox_score[0]), int(bbox_score[1]),
                    int(bbox_score[2] - bbox_score[0]), int(bbox_score[3] - bbox_score[1])]
            score = bbox_score[4]
            cls_id = CLASS_NAME.index(cls_name) + 1
            outputs.append({
                "bbox": bbox,
                "score": score,
                "image_id": img_id,
                "category_id": cls_id
            })

def vis(img, results, conf=0.5, class_names=None):
    for cls_name in results:
        for bbox_score in results[cls_name]:
            x0, y0, x1, y1, score = bbox_score
            cls_id = class_names.index(cls_name)
            if score < conf:
                continue
            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)

            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (int(x0), int(y0 + 1)),
                (int(x0 + txt_size[0] + 1), int(y0 + 1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (int(x0), int(y0 + txt_size[1])), font, 0.4, txt_color, thickness=1)
    return img


def get_file_list(img_dir, test_dir):
    json_file = os.path.join(test_dir, "annotations.json")

    if os.path.exists(json_file):
        coco_gt = COCO(json_file)
        imgInfo = coco_gt.loadImgs(coco_gt.getImgIds())
        file_list = [img_info["file_name"] for img_info in imgInfo]
    else:
        file_list = os.listdir(img_dir)
        file_list.sort()
    
    return file_list

if __name__ == "__main__":
    args = parser.parse_args()
    test_dir = args.test_dir
    model_dir = args.model_dir

    predictor = EnsamblePredictor()

    if predictor.loadModel(model_dir):
        img_dir = os.path.join(test_dir, "image")
        file_list = get_file_list(img_dir, test_dir)
        T = 0
        current_time = time.localtime()
        outputs = []
        for img_id, img_name in enumerate(file_list):
            t0 = time.time()
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            results = predictor.predict(img)
            if args.save_result:
                save_folder = os.path.join(test_dir, "vis_res")
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder, exist_ok=True)
                visual(img_path, img, save_folder, results, current_time)
            save_results(img_id, results, outputs)
            t1 = time.time()
            T += (t1 - t0)
            logging.info("{}/{}, Infer time: {:.4f}ms".format((img_id + 1), len(file_list), (t1 - t0) * 1000))
        result_filename = os.path.join(test_dir, "ensamble_results.json")
        with open(result_filename, 'w') as jf:
            json.dump(outputs, jf, indent=4)
        logging.info("mT: {}".format(T / len(file_list)))
