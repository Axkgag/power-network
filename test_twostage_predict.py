import argparse
import os
import time
import cv2
import numpy as np
import json
import logging

from copy import deepcopy
from pycocotools.coco import COCO
from task.object_detection.yolox import yolox_predictor
# from task.object_detection.centernet import centernet_predictor
from task.two_stage.obj_cls import twostage_predictor
# from task.classification.effnet import effnet_predictor
from task.classification.mulnet import mulnet_predictor
from utils.riemann_coco_eval import getAPForRes

parser = argparse.ArgumentParser()
# parser.add_argument('-p', '--predictor_dir',
#                     default='../../ChipCounter/data/chip/datasets/wildanimals/train/yolox/export/',
#                     help='model_dir, dir of model to be loaded')
# parser.add_argument('-c', '--classifier_dir',
#                     default='../../ChipCounter/data/chip/datasets/wildanimals/train/effnet/export/',
#                     help='model_dir, dir of model to be loaded')
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
    # img = cv2.imread(img_dir)
    # img = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), -1)
    result_image = vis(img, results, 0.5, CLASS_NAME)

    save_folder = os.path.join(
        save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    save_file_name = os.path.join(save_folder, os.path.basename(img_path))
    # logging.info("Saving detection result in {}".format(save_file_name))
    # cv2.imwrite(save_file_name, result_image)
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

# def get_file_list(test_dir):
#     json_file = os.path.join(test_dir, "annotations.json")
#     coco_gt = COCO(json_file)
#     imgInfo = coco_gt.loadImgs(coco_gt.getImgIds())
#     file_list = [img_info["file_name"] for img_info in imgInfo]
#     return file_list

if __name__ == "__main__":
    args = parser.parse_args()
    test_dir = args.test_dir
    model_dir = args.model_dir

    predictor = yolox_predictor.createInstance(config_path=os.path.join(model_dir, "stage1/predictor.json"))
    # classifier = effnet_predictor.createInstance(config_path=os.path.join(model_dir, "stage2/predictor.json"))

    # predictor = centernet_predictor.createInstance(config_path=os.path.join(model_dir, "stage1/predictor.json"))
    classifier = mulnet_predictor.createInstance(config_path=os.path.join(model_dir, "stage2/predictor.json"))

    # predictor.loadModel(predictor_dir)
    # classifier.loadModel(classifier_dir)

    twostage = twostage_predictor.createInstance(predictor, classifier)
    twostage.loadModel(model_dir)

    img_dir = os.path.join(test_dir, "image")
    file_list = os.listdir(img_dir)
    file_list.sort()
    # file_list = get_file_list(test_dir)
    T = 0
    current_time = time.localtime()
    outputs = []
    for img_id, img_name in enumerate(file_list):
        t0 = time.time()
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        results = twostage.predict(img)
        if args.save_result:
            save_folder = os.path.join(test_dir, "vis_res")
            if not os.path.exists:
                os.makedirs(save_folder, exist_ok=True)
            # visual(img_path, img, save_folder, results, current_time)
        save_results(img_id, results, outputs)
        t1 = time.time()
        T += (t1 - t0)
        logging.info("{}/{}, Infer time: {:.4f}ms".format((img_id + 1), len(file_list), (t1 - t0) * 1000))
    # if args.save_result:
    result_filename = os.path.join(test_dir, "twostage_results.json")
    origin_filename = os.path.join(test_dir, "annotations.json")
    with open(result_filename, 'w') as jf:
        json.dump(outputs, jf, indent=4)
        result = getAPForRes(result_filename, origin_filename, "bbox")
        # print("result:", result)
    print("mT: ", T / len(file_list))
