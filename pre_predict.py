import argparse
import os
import time
import cv2
import numpy as np
import json
import logging

import xml.etree.ElementTree as ET
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
                    default='./model',
                    help='test_dir, dir of dataset be tested')
parser.add_argument('-t', '--test_dir',
                    default='/data/gauss/lyh/datasets/power_networks/test/',
                    help='test_dir, dir of dataset be tested')
parser.add_argument('-s', '--save_result',
                    default=False,
                    action='store_true',
                    help='save result or not')

CLASS_NAME = {
        "010101021": "纵横向裂纹",
        "010101031": "法兰杆法兰锈蚀",
        "010101061": "异物鸟巢",
        "010101071": "塔顶损坏",
        "010201041": "绝缘线绝缘层破损",
        "010202011": "扎绑带不规范",
        "010202021": "扎绑带未扎绑",
        "010301021": "绝缘子破损",
        "010301011": "绝缘子污秽",
        "010301031": "绝缘子固定不牢固",
        "010401041": "线夹绝缘罩脱落",
        "010401051": "保险销子脱落",
        "010401061": "球头锈蚀",
        "050101041": "熔断器绝缘罩缺失",
        "060101051": "避雷器脱扣"
    }

def create_xml(img_path, results_dict, output_dir, img_info):
    # 创建根节点
    root = ET.Element("annotation")

    # 添加子节点和文本内容
    folder = ET.SubElement(root, "folder")
    folder.text = "image"

    img_name = os.path.basename(img_path)
    filename = ET.SubElement(root, "filename")
    filename.text = img_name

    path = ET.SubElement(root, "path")
    path.text = img_path

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    H, W, C = img_info
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(W)
    height = ET.SubElement(size, "height")
    height.text = str(H)
    depth = ET.SubElement(size, "depth")
    depth.text = str(C)

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    results = results_dict["results"]

    if not results:
        return

    for cls_id in results:
        for bbox_score in results[cls_id]:
            cls_name = CLASS_NAME[cls_id]    

            object_ = ET.SubElement(root, "object")

            name = ET.SubElement(object_, "name")
            name.text = cls_name

            pose = ET.SubElement(object_, "pose")
            pose.text = "Unspecified"

            truncated = ET.SubElement(object_, "truncated")
            truncated.text = "0"

            difficult = ET.SubElement(object_, "difficult")
            difficult.text = "0"

            bndbox = ET.SubElement(object_, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(int(bbox_score[0]))
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(int(bbox_score[1]))
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(bbox_score[2]))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(bbox_score[3]))

    # 创建 XML 树
    tree = ET.ElementTree(root)
    # 将 XML 树写入文件
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    outputs = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".xml")
    tree.write(outputs, encoding="utf-8")


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
        try:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            results = twostage.predict(img)
        except:
            logging.error("{} predict fail".format(img_name))
            continue
        # if args.save_result:
        #     save_folder = os.path.join(test_dir, "vis_res")
        #     if not os.path.exists:
        #         os.makedirs(save_folder, exist_ok=True)
        #     visual(img_path, img, save_folder, results, current_time)
        # save_results(img_id, results, outputs)
        create_xml(img_path, results, os.path.join(test_dir, "label"), img.shape)
        # print(results["results"])
        t1 = time.time()
        T += (t1 - t0)
        logging.info("{}/{} finished detection {}, Time:{}".format(img_id + 1, len(file_list), img_name, t1 - t0))
    # if args.save_result:
    result_filename = os.path.join(test_dir, "results.json")
    origin_filename = os.path.join(test_dir, "annotations.json")
    # with open(result_filename, 'w') as jf:
    #     json.dump(outputs, jf, indent=4)
        # result = getAPForRes(result_filename, origin_filename, "bbox")
        # print("result:", result)
    print("mT: ", T / len(file_list))
