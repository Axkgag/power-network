import os
import math
import numpy as np
import json
import pycocotools.coco as coco
from dotadevkit.polyiou import polyiou

def getAPForRes(res_filename, origin_filename, save_dir):
    cache_dir = os.path.join(save_dir, "cache")
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    txt_save_dir = os.path.join(cache_dir, "txt_res")
    gt_save_dir = os.path.join(cache_dir, "txt_gt")

    imgs_list, cats_list = load_coco(origin_filename, cache_dir)
    load_results(res_filename, cache_dir, imgs_list, cats_list)

    map = 0
    classaps = []
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
        classaps.append({
            "class_name": cls,
            "ap": ap
        })
    map /= len(cats_list)
    results = {
        "map": map,
        "class_ap_list": classaps
    }
    return results

def convert_box(bbox):
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
        x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = convert_box(bbox)
        converted_res[cls_name].append(
            img_name.split(".")[0] + " " + str(score) + " " +
            str(x0n) + " " + str(y0n) + " " + str(x1n) + " " +
            str(y1n) + " " + str(x2n) + " " + str(y2n) + " " +
            str(x3n) + " " + str(y3n) + "\n")
    for cls in cats_list:
        with open(os.path.join(save_dir, cls+'.txt'), 'w') as wf:
            wf.writelines(converted_res[cls])

def load_coco(json_dir, save_dir):
    coco_anno = coco.COCO(json_dir)
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
            x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = convert_box(bbox)
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

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,   # txt_predict/class_name.txt
             annopath,  # txt_gt/img_name.txt
             imagesetfile,  # valset.txt
             classname,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """

    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_gt(annopath.format(imagename))
        # annopath.format(imagename) = ./txt_gt/img_name.txt

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    if(len(BB)==0): return 0, 0, 0

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    # print('check fp:', fp)
    # print('check tp', tp)
    #
    #
    # print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
