import os
import math
import json
import argparse
import numpy as np
from pycocotools import mask
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

np.set_printoptions(linewidth=500)

cls_bins = {
        "010101021": "rare",
        "010101031": "rare",
        "010101061": "normal",
        "010101071": "normal",
        "010201041": "frequent",
        "010202021": "rare",
        "010301021": "rare",
        "010301011": "normal",
        "010301031": "normal",
        "010401041": "frequent",
        "010401051": "frequent",
        "010401061": "normal",
        "050101041": "frequent",
        "060101051": "normal"
    }

def cal_iou(bbox1, bbox2):
    area = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3]

    x11 = max(bbox1[0], bbox2[0])
    y11 = max(bbox1[1], bbox2[1])
    x22 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y22 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    w, h = max(0, x22 - x11 + 1), max(0, y22 - y11 + 1)
    overlap = w * h
    iou = overlap / (area - overlap + 1e-16)

    return iou


def eval(gt_file, dt_file, iou_threshold, mode="obj"):
    results = {
        "map": 0,
        "all_bbox": {},
        "size":{},
        "frequency":{},
        "categories": {}
    }

    # 创建COCO API对象
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(dt_file)

    # 获取图片列表
    img_ids = coco_gt.getImgIds()

    # 获取类别列表
    cat_ids = coco_gt.getCatIds()
    num_classes = len(cat_ids)
    cats = coco_gt.loadCats(cat_ids)
    cat_names = [cat['name'] for cat in cats]
    for cat_name in cat_names:
        results["categories"][cat_name] = {}

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map = coco_eval.stats.tolist()[:6]
    # results["map"] = round(map, 3)
    results["map"] = map

    for cat in cats:
        coco_eval.params.catIds = cat["id"]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        clas_ap = coco_eval.stats.tolist()[:6]
        results["categories"][cat["name"]]["ap"] = clas_ap[:6]

    
    # 初始化指标计数器: '_b' for box
    tp_b = {cat: 0 for cat in cat_names}
    fp_b = {cat: 0 for cat in cat_names}
    fn_b = {cat: 0 for cat in cat_names}

    tp_s = 0
    fp_s = 0
    fn_s = 0

    tp_l = 0
    fp_l = 0
    fn_l = 0

    tp_f = 0
    fp_f = 0
    fn_f = 0

    tp_n = 0
    fp_n = 0
    fn_n = 0

    tp_r = 0
    fp_r = 0
    fn_r = 0

    nums_s = 0
    nums_l = 0


    # 遍历每个图片的gt
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)
        img_width = img_info[0]["width"]
        img_height = img_info[0]["height"]
        img_area = img_height * img_width

        gt_annos = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        dt_annos = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))

        # 计算gt bbox指标
        for gt_ann in gt_annos:
            cat_id = gt_ann['category_id']
            cat_name = coco_gt.loadCats(cat_id)[0]['name']

            mode_f = cls_bins[cat_name] 

            gt = gt_ann["bbox"]
            gt_area = gt_ann["area"]

            mode_s = "small" if (gt_area * 100 / img_area) <= 1 else "large" 
            if mode_s == "small":
                nums_s += 1
            elif mode_s == "large":
                nums_l += 1

            match = False  

            for dt_ann in dt_annos:
                dt = dt_ann['bbox']

                if cal_iou(dt, gt) > iou_threshold:
                    if dt_ann['category_id'] == cat_id:
                        match = True
                        break

            if match:
                tp_b[cat_name] += 1

                if mode_f == 'rare':
                    tp_r += 1
                elif mode_f == 'normal':
                    tp_n += 1
                elif mode_f == 'frequent':
                    tp_f += 1

                if mode_s == "small":
                    tp_s += 1
                elif mode_s == "large":
                    tp_l += 1
            else:
                fn_b[cat_name] += 1

                if mode_f == 'rare':
                    fn_r += 1
                elif mode_f == 'normal':
                    fn_n += 1
                elif mode_f == 'frequent':
                    fn_f += 1

                if mode_s == "small":
                    fn_s += 1
                elif mode_s == "large":
                    fn_l += 1
            
        # 计算dt bbox指标
        for dt_ann in dt_annos:
            cat_id = dt_ann["category_id"]
            cat_name = coco_gt.loadCats(cat_id)[0]['name']

            mode_f = cls_bins[cat_name] 

            dt = dt_ann['bbox']

            match = False

            for gt_ann in gt_annos:
                gt = gt_ann["bbox"]
                gt_area = gt_ann['area']

                mode_s = "small" if (gt_area * 100 / img_area) <= 1 else "large" 

                if cal_iou(gt, dt) > iou_threshold:
                    
                    if gt_ann["category_id"] == cat_id:
                        match = True
                        break

            if not match:
                fp_b[cat_name] += 1

                if mode_f == 'rare':
                    fp_r += 1
                elif mode_f == 'normal':
                    fp_n += 1
                elif mode_f == 'frequent':
                    fp_f += 1

                if mode_s == "small":
                    fp_s += 1
                elif mode_s == "large":
                    fp_l += 1


    # plot(ConfusionMatrix)

    # 计算评价指标
    recall_b = sum(tp_b.values()) / (sum(tp_b.values()) + sum(fn_b.values()) + 1e-16)
    error_b = (sum(fn_b.values()) + sum(fp_b.values())) / (sum(tp_b.values()) + sum(fn_b.values()) + 1e-16)

    results['all_bbox'] = {
        "detection_rate": round(recall_b, 3),
        "false_detection_ratio": round(error_b, 3)
    }

    recall_s = tp_s / (tp_s + fn_s + 1e-16)
    error_s = (fn_s + fp_s) / (tp_s + fn_s + 1e-16)

    recall_l = tp_l / (tp_l + fn_l + 1e-16)
    error_l = (fn_l + fp_l) / (tp_l + fn_l + 1e-16)

    results['size'] = {
        "small":{
            "nums": nums_s,
            "detection_rate": round(recall_s, 3),
            "false_detection_ratio": round(error_s, 3)
        },
        "large": {
            "nums": nums_l,
            "detection_rate": round(recall_l, 3),
            "false_detection_ratio": round(error_l, 3)
        }
    }


    recall_f = tp_f / (tp_f + fn_f + 1e-16)
    error_f = (fn_f + fp_f) / (tp_f + fn_f + 1e-16)

    recall_n = tp_n / (tp_n + fn_n + 1e-16)
    error_n = (fn_n + fp_n) / (tp_n + fn_n + 1e-16)

    recall_r = tp_r / (tp_r + fn_r + 1e-16)
    error_r = (fn_r + fp_r) / (tp_r + fn_r + 1e-16)

    results['frequency'] = {
        "rare":{
            "detection_rate": round(recall_r, 3),
            "false_detection_ratio": round(error_r, 3)
        },
        "normal": {
            "detection_rate": round(recall_n, 3),
            "false_detection_ratio": round(error_n, 3)
        },
        "frequent": {
            "detection_rate": round(recall_f, 3),
            "false_detection_ratio": round(error_f, 3)
        }
    }

    for cat in cat_names:
        recall = tp_b[cat] / (tp_b[cat] + fn_b[cat] + 1e-16)
        error = (fn_b[cat] + fp_b[cat]) / (tp_b[cat] + fn_b[cat] + 1e-16)

        results["categories"][cat]["bbox nums"] = tp_b[cat] + fn_b[cat]
        results["categories"][cat]["detection_rate"] = round(recall, 3)
        results["categories"][cat]["false_detection_ratio"] = round(error, 3)

    print(results)
    return results


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--ann_dir',
                    default="./datasets/grid/annotations/test_annotations.json",
                    help='annotations.json file dir')
parser.add_argument('-r', '--res_dir',
                    default="./outputs/HR_50_2/test_results.json",
                    help='results.json file dir')
parser.add_argument('-t', '--iou_thres',
                    default=0.4,
                    type=float,
                    help='iou threshold')
parser.add_argument('-m', '--mode',
                    default="obj",
                    help='obj/rot/def/seg')
parser.add_argument('-s', '--save_path',
                    default="./outputs/HR_50_2/eval2.json",
                    help='eval results save path')


if __name__ == "__main__":
    # 加载测试集中的图像标签和推理结果
    args = parser.parse_args()
    gt_file = args.ann_dir
    dt_file = args.res_dir
    iou_threshold = args.iou_thres
    mode = args.mode
    save_path = args.save_path

    results = eval(gt_file, dt_file, iou_threshold, mode)
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)



