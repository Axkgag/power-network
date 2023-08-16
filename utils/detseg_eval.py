import os
import math

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from dotadevkit.polyiou import polyiou

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

def cal_iou(bbox1, bbox2):
    area = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3]

    x11 = max(bbox1[0], bbox2[0])
    y11 = max(bbox1[1], bbox2[1])
    x22 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y22 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    if len(bbox1) == 4:
        w, h = max(0, x22 - x11 + 1), max(0, y22 - y11 + 1)
        overlap = w * h
        iou = overlap / (area - overlap + 1e-16)
    elif len(bbox1) == 5:
        rbbox1 = convert_bbox(bbox1)
        rbbox2 = convert_bbox(bbox2)
        iou = polyiou.iou_poly(polyiou.VectorDouble(rbbox1), polyiou.VectorDouble(rbbox2))

    return iou

def eval(gt_file, dt_file, iou_threshold):
    # 创建COCO API对象
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(dt_file)

    # coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()

    # 获取图片列表
    img_ids = coco_gt.getImgIds()

    # 获取类别列表
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    cat_names = [cat['name'] for cat in cats]

    # 初始化指标计数器: '_b' for box, '_p' for picture
    tp_b = {cat: 0 for cat in cat_names}
    fp_b = {cat: 0 for cat in cat_names}
    fn_b = {cat: 0 for cat in cat_names}

    tp_p = fp_p = fn_p = tn_p = 0

    # 遍历每个图片的gt
    for img_id in img_ids:
        gt_annos = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        dt_annos = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))

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

        for gt_ann in gt_annos:
            cat_id = gt_ann['category_id']
            cat_name = coco_gt.loadCats(cat_id)[0]['name']
            gt_bbox = gt_ann['bbox']

            gt_match = False

            for dt_ann in dt_annos:
                if dt_ann['category_id'] == cat_id:
                    dt_bbox = dt_ann['bbox']

                    if cal_iou(dt_bbox, gt_bbox) > iou_threshold:
                        gt_match = True
                        break
            if gt_match:
                tp_b[cat_name] += 1
            else:
                fn_b[cat_name] += 1

        for dt_ann in dt_annos:
            cat_id = dt_ann["category_id"]
            cat_name = coco_gt.loadCats(cat_id)[0]['name']
            dt_bbox = dt_ann['bbox']

            dt_match = False

            for gt_ann in gt_annos:
                if gt_ann["category_id"] == cat_id:
                    gt_bbox = gt_ann["bbox"]

                    if cal_iou(gt_bbox, dt_bbox) > iou_threshold:
                        dt_match = True
                        break
            if not dt_match:
                fp_b[cat_name] += 1

    # 计算评价指标
    # mAP = sum(tp.values()) / (sum(tp.values()) + sum(fp.values()) + sum(fn.values()))
    # precision_b = sum(tp_b.values()) / (sum(tp_b.values()) + sum(fp_b.values()))
    recall_b = sum(tp_b.values()) / (sum(tp_b.values()) + sum(fn_b.values()))
    loss_b = sum(fn_b.values()) / (sum(tp_b.values()) + sum(fn_b.values()))
    error_b = (sum(fn_b.values()) + sum(fp_b.values())) / (sum(tp_b.values()) + sum(fn_b.values()))

    print("----for box----")
    # print("mAP: {:.3f}".format(precision_b))
    print("发现率: {:.3f}".format(recall_b))
    print("漏检率: {:.3f}".format(loss_b))
    print("误检比: {:.3f}".format(error_b))

    for cat in cat_names:
        print("----class: {}----".format(cat))
        recall = tp_b[cat] / (tp_b[cat] + fn_b[cat])
        loss = fn_b[cat] / (tp_b[cat] + fn_b[cat])
        error = (fn_b[cat] + fp_b[cat]) / (tp_b[cat] + fn_b[cat])
        print("发现率: {:.3f}".format(recall))
        print("漏检率: {:.3f}".format(loss))
        print("误检比: {:.3f}".format(error))

    recall_p = (tp_p + tn_p) / (tp_p + tn_p + fn_p + fp_p + 1e-16)
    loss_p = fp_p / (fp_p + tn_p + 1e-16)
    error_p = fn_p / (tp_p + fn_p + 1e-16)

    print("----for picture----")
    print("发现率: {:.3f}".format(recall_p))
    print("漏检率: {:.3f}".format(loss_p))
    print("误检比: {:.3f}".format(error_p))

def main():
    # 加载测试集中的图像标签和推理结果
    data_dir = r"D:\task\ChipCounter\data\chip\23-02-06\yoloxr\no_patch\test"
    gt_file = os.path.join(data_dir, "annotations.json")
    dt_file = os.path.join(data_dir, "results.json")

    # 参数设定
    iou_threshold = 0.5

    eval(gt_file, dt_file, iou_threshold)


if __name__ == "__main__":
    main()



