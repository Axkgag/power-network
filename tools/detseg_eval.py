import os
import math
import json
import argparse
import numpy as np
from pycocotools import mask
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from dotadevkit.polyiou import polyiou
from dotadevkit.evaluate.task1 import voc_eval

np.set_printoptions(linewidth=500)

def convert_bbox(bbox):
    x, y, width, height = int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2), int(bbox[2]), int(bbox[3])
    anglePi = bbox[4] / 180 * math.pi
    anglePi = anglePi if anglePi <= math.pi else anglePi - math.pi

    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)

    x1 = x - 0.5 * width
    y1 = y - 0.5 * height

    x0 = x + 0.5 * width
    y0 = y1

    x2 = x1
    y2 = y + 0.5 * height

    x3 = x0
    y3 = y2

    x0n = int((x0 - x) * cosA - (y0 - y) * sinA + x)
    y0n = int((x0 - x) * sinA + (y0 - y) * cosA + y)

    x1n = int((x1 - x) * cosA - (y1 - y) * sinA + x)
    y1n = int((x1 - x) * sinA + (y1 - y) * cosA + y)

    x2n = int((x2 - x) * cosA - (y2 - y) * sinA + x)
    y2n = int((x2 - x) * sinA + (y2 - y) * cosA + y)

    x3n = int((x3 - x) * cosA - (y3 - y) * sinA + x)
    y3n = int((x3 - x) * sinA + (y3 - y) * cosA + y)
    return x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n

def load_results(json_dir, save_dir, imgs_list, cats_list):
    save_dir = os.path.join(save_dir, 'txt_res')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(json_dir, 'r') as jf:
        data = json.load(jf)
    converted_res = {}
    for cls in cats_list:
        converted_res[cls] = []
    for anno in data:
        bbox = anno['bbox']
        cat_id = anno['category_id']
        cls_name = cats_list[cat_id - 1]
        score = anno['score']
        img_id = anno['image_id']
        img_name = imgs_list[img_id]
        x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = convert_bbox(bbox)
        converted_res[cls_name].append(
            img_name.split(".")[0] + " " + str(score) + " " +
            str(x0n) + " " + str(y0n) + " " + str(x1n) + " " +
            str(y1n) + " " + str(x2n) + " " + str(y2n) + " " +
            str(x3n) + " " + str(y3n) + "\n")
    for cls in cats_list:
        with open(os.path.join(save_dir, cls+'.txt'), 'w') as wf:
            wf.writelines(converted_res[cls])

def load_coco(json_dir, save_dir):
    coco_anno = COCO(json_dir)
    img_ids = coco_anno.getImgIds()
    cat_ids = coco_anno.getCatIds()

    imgs_list = []
    cats_list = []
    converted_res = {}

    for img_info in coco_anno.loadImgs(img_ids):
        imgs_list.append(img_info['file_name'])
        converted_res[img_info['file_name'].split(".")[0]] = []
    for cat_info in coco_anno.loadCats(cat_ids):
        cats_list.append(cat_info['name'])
    for id, img_name in enumerate(imgs_list):
        anns = coco_anno.loadAnns(coco_anno.getAnnIds(imgIds=id))
        img = img_name.split(".")[0]
        for ann in anns:
            cat_id = ann['category_id']
            cat_name = cats_list[cat_id - 1]
            bbox = ann['bbox']
            x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = convert_bbox(bbox)
            converted_res[img].append(
                str(x0n) + " " + str(y0n) + " " + str(x1n) + " " +
                str(y1n) + " " + str(x2n) + " " + str(y2n) + " " +
                str(x3n) + " " + str(y3n) + " " + cat_name + " " + "0" + "\n")

    with open(os.path.join(save_dir, "trainset.txt"), "w") as f1:
        for name in imgs_list:
            f1.write(name.split(".")[0] + " " + "\n")
    gt_save_dir = os.path.join(save_dir, "txt_gt")
    if not os.path.exists(gt_save_dir):
        os.mkdir(gt_save_dir)
    for img in imgs_list:
        img = img.split(".")[0]
        with open(os.path.join(gt_save_dir, img + ".txt"), 'w') as f2:
            f2.writelines(converted_res[img])
    return imgs_list, cats_list

def parse_gt(filename):
    """
    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    # filename = ./txt_gt/img_name.txt
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if len(splitlines) < 9:
                    continue
                object_struct['name'] = splitlines[8]

                if len(splitlines) == 9:
                    object_struct['difficult'] = 0
                elif len(splitlines) == 10:
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects

def cal_iou(bbox1, bbox2, mode="obj"):
    if mode == "obj" or mode == "def":
        area = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3]

        x11 = max(bbox1[0], bbox2[0])
        y11 = max(bbox1[1], bbox2[1])
        x22 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y22 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

        w, h = max(0, x22 - x11 + 1), max(0, y22 - y11 + 1)
        overlap = w * h
        iou = overlap / (area - overlap + 1e-16)
    elif mode == "rot":
        rbbox1 = convert_bbox(bbox1)
        rbbox2 = convert_bbox(bbox2)
        iou = polyiou.iou_poly(polyiou.VectorDouble(rbbox1), polyiou.VectorDouble(rbbox2))
    elif mode == "seg":
        mask1 = mask.decode(bbox1)
        mask2 = mask.decode(bbox2)

        area1 = mask1.sum()
        area2 = mask2.sum()
        inter = ((mask1 + mask2) == 2).sum()
        iou = inter / (area1 + area2 - inter)

    return iou

def plot(ConfusionMatrix, normalize=False, names=None):
    import seaborn as sn
    import matplotlib.pyplot as plt
    import warnings
    import copy
    # raw_matrix = np.array(raw_matrix)
    # n = raw_matrix.shape[0]
    # matrix = np.zeros((n + 1, n + 1))
    # matrix[: n, : n] = raw_matrix

    # diagonal_elements = np.diagonal(raw_matrix)
    # row_sums = np.sum(raw_matrix, axis=1)
    # col_sums = np.sum(raw_matrix, axis=0)

    # matrix[n, : n] = diagonal_elements / col_sums
    # matrix[:n, n] = diagonal_elements / row_sums
    matrix = copy.deepcopy(ConfusionMatrix)
    matrix[matrix < 0.005] = np.nan

    num_classes = len(matrix[0]) - 2
    class_names = names if names else [str(id + 1) for id in range(num_classes)] 
    # class_names += ["background", "rate"]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    sn.set(font_scale=1.0 if num_classes < 50 else 0.8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sn.heatmap(
            matrix,
            ax=ax,
            annot=num_classes < 30,
            annot_kws={
                "size": 8
            },
            cmap="Blues",
            fmt='.3f',
            square=True,
            vmin=0.0,
            xticklabels=class_names + ["BG", "CR"],
            yticklabels=class_names + ["BG", "DR"]
        ).set_facecolor((1, 1, 1))
    ax.set_xlabel("Ground True")
    ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix")
    fig.savefig('confusion_matrix.png', dpi=250)
    plt.close(fig)

def eval(gt_file, dt_file, iou_threshold, mode="obj"):
    results = {
        "map": 0,
        "all_bbox": {},
        "all_bbox_ignore_category": {},
        "categories": {}
    }

    gt_sum = dt_sum = 0

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

    # 初始化混淆矩阵
    ConfusionMatrix = np.zeros((num_classes + 2, num_classes + 2))

    # 计算mAP
    if mode == 'obj' or mode == 'def':
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map = coco_eval.stats.tolist()[0]
        results["map"] = round(map, 3)

        for cat in cats:
            coco_eval.params.catIds = cat["id"]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            clas_ap = coco_eval.stats.tolist()[0]
            # results["classAPs"].append({
            #     "class_name": cat["name"],
            #     "AP": round(clas_ap, 3)
            # })
            results["categories"][cat["name"]]["ap"] = round(clas_ap, 3)

    elif mode == 'seg':
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map = coco_eval.stats.tolist()[0]
        results["map"] = round(map, 3)

        for cat in cats:
            coco_eval.params.catIds = cat["id"]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            clas_ap = coco_eval.stats.tolist()[0]
            # results["classAPs"].append({
            #     "class_name": cat["name"],
            #     "AP": round(clas_ap, 3)
            # })
            results["categories"][cat["name"]]["ap"] = round(clas_ap, 3)

    elif mode == "rot":
        cache_dir = os.path.dirname(gt_file) + '/cache'
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        txt_save_dir = os.path.join(cache_dir, "txt_res")
        gt_save_dir = os.path.join(cache_dir, "txt_gt")
        
        imgs_list, cats_list = load_coco(gt_file, cache_dir)
        load_results(dt_file, cache_dir, imgs_list, cats_list)
        
        map = 0
        # classaps = []
        for cls in cats_list:
            rec, prec, ap = voc_eval(
                txt_save_dir + r'/{:s}.txt',
                gt_save_dir + r'/{:s}.txt',
                cache_dir + r'/trainset.txt',
                cls,
                ovthresh=0.5,
                use_07_metric=True
            )
            map += ap
            # classaps.append({
            #     "class_name": cls,
            #     "ap": ap
            # })
            # results['classAPs'].append({
            #     "class_name": cls,
            #     "ap": round(ap, 3)
            # })
            results["categories"][cls]["ap"] = round(ap, 3)
        map /= len(cats_list)
        results['map'] = round(map, 3)
    
    # 初始化指标计数器: '_b' for box, '_p' for picture, '_nc' for ignore categories
    tp_b = {cat: 0 for cat in cat_names}
    fp_b = {cat: 0 for cat in cat_names}
    fn_b = {cat: 0 for cat in cat_names}
    tp_nc = fp_nc = fn_nc = 0

    tp_p = fp_p = fn_p = tn_p = 0

    # 遍历每个图片的gt
    for img_id in img_ids:
        gt_annos = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        dt_annos = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))

        gt_sum += len(coco_gt.getAnnIds(imgIds=img_id))
        dt_sum += len(coco_dt.getAnnIds(imgIds=img_id))

        # 计算图片指标
        gt_nums = len(gt_annos)
        dt_nums = len(dt_annos)
        if gt_nums == 0 and dt_nums == 0:
            tp_p += 1
        elif gt_nums > 0 and dt_nums > 0:
            tn_p += 1
        elif gt_nums == 0 and dt_nums > 0:
            fn_p += 1
        elif gt_nums > 0 and dt_nums == 0:
            fp_p += 1

        # 计算gt bbox指标
        for gt_ann in gt_annos:
            cat_id = gt_ann['category_id']
            cat_name = coco_gt.loadCats(cat_id)[0]['name']
            # gt_bbox = gt_ann['bbox']
            # gt_segm = gt_ann['segmentation']

            gt = gt_ann['segmentation'] if mode == 'seg' else gt_ann["bbox"]

            gt_match = False  # 关注类别和IoU
            match = False  # 仅关注IoU

            for dt_ann in dt_annos:
                dt_bbox = dt_ann['bbox']
                dt_segm = dt_ann['segmentation']

                dt = dt_ann['segmentation'] if mode == 'seg' else dt_ann['bbox']

                # if mode == 'seg':
                #     gt = gt_segm
                #     dt = dt_segm
                # else:
                #     gt = gt_bbox
                #     dt = dt_bbox

                if cal_iou(dt, gt, mode) > iou_threshold:
                    match = True
                    if dt_ann['category_id'] == cat_id:
                        gt_match = True
                        ConfusionMatrix[cat_id - 1][cat_id - 1] += 1  # 检测正确
                        break
                    else:
                        ConfusionMatrix[dt_ann['category_id'] - 1][cat_id - 1] += 1  # 类别检测错误，但IoU达标（错检）

            if gt_match:
                tp_b[cat_name] += 1
            else:
                fn_b[cat_name] += 1
            
            if match:
                tp_nc += 1
            else:
                fn_nc += 1
                ConfusionMatrix[-2][cat_id - 1] += 1  # 不存在IoU和类别匹配的dt框（漏检）

        # 计算dt bbox指标
        for dt_ann in dt_annos:
            cat_id = dt_ann["category_id"]
            cat_name = coco_gt.loadCats(cat_id)[0]['name']
            # dt_bbox = dt_ann['bbox']
            # dt_segm = dt_ann['segmentation']

            dt = dt_ann['segmentation'] if mode == 'seg' else dt_ann['bbox']

            dt_match = False
            match = False

            for gt_ann in gt_annos:
                # gt_bbox = gt_ann['bbox']
                # gt_segm = gt_ann['segmentation']

                gt = gt_ann['segmentation'] if mode == 'seg' else gt_ann["bbox"]

                # if mode == 'seg':
                #     gt = gt_segm
                #     dt = dt_segm
                # else:
                #     gt = gt_bbox
                #     dt = dt_bbox

                if cal_iou(gt, dt, mode) > iou_threshold:
                    match = True
                    if gt_ann["category_id"] == cat_id:
                        dt_match = True
                        break

            if not dt_match:
                fp_b[cat_name] += 1
            if not match:
                fp_nc += 1
                ConfusionMatrix[cat_id - 1][-2] += 1

    n = ConfusionMatrix.shape[0] - 1

    diagonal_elements = np.diagonal(ConfusionMatrix)[:n]
    row_sums = np.sum(ConfusionMatrix, axis=1)[:n]
    col_sums = np.sum(ConfusionMatrix, axis=0)[:n]

    ConfusionMatrix[n, : n] = np.round(diagonal_elements / (col_sums + 1e-16), 2)
    ConfusionMatrix[:n, n] = np.round(diagonal_elements / (row_sums + + 1e-16), 2)

    # plot(ConfusionMatrix)

    # 计算评价指标
    recall_b = sum(tp_b.values()) / (sum(tp_b.values()) + sum(fn_b.values()) + 1e-16)
    loss_b = sum(fn_b.values()) / (sum(tp_b.values()) + sum(fn_b.values()) + 1e-16)
    error_b = (sum(fn_b.values()) + sum(fp_b.values())) / (sum(tp_b.values()) + sum(fn_b.values()) + 1e-16)
    error_rate_b = (sum(fn_b.values()) + sum(fp_b.values())) / (sum(tp_b.values()) + sum(fn_b.values()) + sum(fp_b.values()) + 1e-16)

    results['all_bbox'] = {
        "detection_rate": round(recall_b, 3),
        "miss_rate": round(loss_b, 3),
        "false_detection_rate": round(error_rate_b, 3),
        "false_detection_ratio": round(error_b, 3)
    }

    # print("----for all box(categories)----")
    # print("发现率: {:.3f}".format(recall_b))
    # print("漏检率: {:.3f}".format(loss_b))
    # print("误检率: {:.3f}".format(error_rate_b))
    # print("误检比: {:.3f}".format(error_b))

    recall_nc = tp_nc / (tp_nc + fn_nc + 1e-16)
    loss_nc = fn_nc / (tp_nc + fn_nc + 1e-16)
    error_nc = (fn_nc + fp_nc) / (tp_nc + fn_nc + 1e-16)
    error_rate_nc = (fn_nc + fp_nc) / (tp_nc + fn_nc + fp_nc + 1e-16)

    results['all_bbox_ignore_category'] = {
        "detection_rate": round(recall_nc, 3),
        "miss_rate": round(loss_nc, 3),
        "false_detection_rate": round(error_rate_nc, 3),
        "false_detection_ratio": round(error_nc, 3)
    }

    # print("----for all box(ignore categories)----")
    # print("发现率: {:.3f}".format(recall_nc))
    # print("漏检率: {:.3f}".format(loss_nc))
    # print("误检率: {:.3f}".format(error_rate_nc))
    # print("误检比: {:.3f}".format(error_nc))

    for cat in cat_names:
        recall = tp_b[cat] / (tp_b[cat] + fn_b[cat] + 1e-16)
        loss = fn_b[cat] / (tp_b[cat] + fn_b[cat] + 1e-16)
        error = (fn_b[cat] + fp_b[cat]) / (tp_b[cat] + fn_b[cat] + 1e-16)
        error_rate = (fn_b[cat] + fp_b[cat]) / (tp_b[cat] + fn_b[cat] + fp_b[cat] + 1e-16)

        # results['categories'].append({
        #     "class name": cat,
        #     "发现率": round(recall, 3),
        #     "漏检率": round(loss, 3),
        #     "误检率": round(error_rate, 3),
        #     "误检比": round(error, 3)
        # })
        results["categories"][cat]["bbox nums"] = tp_b[cat] + fn_b[cat]
        results["categories"][cat]["detection_rate"] = round(recall, 3)
        results["categories"][cat]["miss_rate"] = round(loss, 3)
        results["categories"][cat]["false_detection_rate"] = round(error_rate, 3)
        results["categories"][cat]["false_detection_ratio"] = round(error, 3)

        # print("----class: {}----".format(cat))
        # print("发现率: {:.3f}".format(recall))
        # print("漏检率: {:.3f}".format(loss))
        # print("误检率: {:.3f}".format(error_rate))
        # print("误检比: {:.3f}".format(error))

    if mode == 'def':
        recall_p = (tp_p + tn_p) / (tp_p + tn_p + fn_p + fp_p + 1e-16)
        loss_p = fp_p / (fp_p + tn_p + 1e-16)
        error_p = fn_p / (tp_p + fn_p + 1e-16)

        results['picture'] = {
            "detection_rate": round(recall_p, 3),
            "miss_rate": round(loss_p, 3),
            "false_detection_rate": round(error_p, 3),
        }

    matrix_list = [np.array2string(row) for row in ConfusionMatrix]
    results["confusionmatrix"] = matrix_list
    # print("----for picture----")
    # print("准确率: {:.3f}".format(recall_p))
    # print("漏检率: {:.3f}".format(loss_p))
    # print("误检率: {:.3f}".format(error_p))

    print(results)
    return results


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--ann_dir',
                    default="../../data/grid/train/annotations.json",
                    help='annotations.json file dir')
parser.add_argument('-r', '--res_dir',
                    default="../../data/grid/train/base_results.json",
                    help='results.json file dir')
parser.add_argument('-t', '--iou_thres',
                    default=0.4,
                    type=float,
                    help='iou threshold')
parser.add_argument('-m', '--mode',
                    default="obj",
                    help='obj/rot/def/seg')
parser.add_argument('-s', '--save_path',
                    # default="../../data/grid/train/rb_eval_results.json",
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



