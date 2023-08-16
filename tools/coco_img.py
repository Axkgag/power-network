import cv2
import numpy as np
import os
import math
import argparse

from pycocotools.coco import COCO

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

def convert_box(bb):
    x, y, width, height = int(bb[0] + bb[2] / 2), int(bb[1] + bb[3] / 2), int(bb[2]), int(bb[3])
    # x, y, width, height = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
    anglePi = bb[4] / 180 * math.pi
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

def make_parser():
    parser = argparse.ArgumentParser("COCO visual")

    parser.add_argument("-r",
                        "--rotation",
                        dest="rotation",
                        default=False,
                        action="store_true",
                        help="rotation or not")

    parser.add_argument("-d",
                        "--data_dir",
                        dest="data_dir",
                        default=None,
                        type=str,
                        help="data dir")
    
    parser.add_argument("-i",
                        "--img_folder",
                        dest="img_folder",
                        default="image",
                        type=str,
                        help="data dir")

    return parser

if __name__ == "__main__":
    args = make_parser().parse_args()

    rotation = args.rotation
    data_dir = args.data_dir
    img_folder = args.img_folder

    json_file = os.path.join(data_dir, "annotations.json")
    img_dir = os.path.join(data_dir, img_folder)
    save_dir = os.path.join(data_dir, "tag")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    coco = COCO(json_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()

    for i in range(len(imgIds)):
        img = coco.loadImgs(imgIds[i])[0]

        # image = cv2.imread(dataset_dir + '/' + img['file_name'])
        image_path = os.path.join(img_dir, img['file_name'])
        image = cv2.imread(image_path)
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        # print(annIds)
        if annIds is None:
            print(img["file_name"])
        annos = coco.loadAnns(annIds)

        if rotation:
            for anno in annos:
                bbox = anno["bbox"]
                cls_id = anno["category_id"]
                cat = coco.loadCats(cls_id)[0]
                cls_name = cat["name"]
                x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = convert_box(bbox)
                text = cls_name
                txt_color = (255, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                # xc, yc = int((bbox_score[0] + bbox_score[2]) / 2), int((bbox_score[1] + bbox_score[3]) / 2)
                # cv2.putText(image, text, (x0n, y0n + txt_size[1]), font, 1, txt_color, thickness=2)

                cv2.line(image, (int(x0n), int(y0n)), (int(x1n), int(y1n)), (0, 0, 255), 2, 4)
                cv2.line(image, (int(x1n), int(y1n)), (int(x2n), int(y2n)), (255, 0, 0), 2, 4)
                cv2.line(image, (int(x2n), int(y2n)), (int(x3n), int(y3n)), (0, 0, 255), 2, 4)
                cv2.line(image, (int(x0n), int(y0n)), (int(x3n), int(y3n)), (255, 0, 0), 2, 4)
        else:
            for anno in annos:
                bbox = anno["bbox"]
                cls_id = anno["category_id"]
                cat = coco.loadCats(cls_id)[0]
                cls_name = cat["name"]
                x, y, w, h = bbox
                color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
                # text = str(COCO_CLASSES[cls_id])
                cv2.rectangle(image,
                              (int(x), int(y)),
                              (int(x + w), int(y + h)),
                              color, 4)

                # text = str(cls_name)
                # txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                #
                # txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                # txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
                # cv2.rectangle(image,
                #               (int(x), int(y + 1)),
                #               (int(x + txt_size[0] + 1), int(y + int(1.5 * txt_size[1]))),
                #               txt_bk_color, -1)
                #
                # cv2.putText(image, text, (int(x), int(y + txt_size[1])), font, 0.4, txt_color, thickness=1)

        save_file_name = save_dir + '/' + img['file_name']
        cv2.imwrite(save_file_name, image)


