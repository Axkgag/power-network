import os
import time
import json
from task.object_detection.yolox import yolox_predictor
import argparse
import logging
import cv2
import numpy as np
from utils.riemann_coco_eval import getAPForRes
from tools import detseg_eval, detseg_eval2

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir',
                    default="",
                    help='model_dir, dir of model to be loaded')
parser.add_argument('-t', '--test_dir',
                    default="./datasets/grid/",
                    help='test_dir, dir of dataset be tested')
parser.add_argument('-s', '--save_result',
                    default=False,
                    action='store_true',
                    help='save result or not')
parser.add_argument('-v', '--val',
                    default=False,
                    action='store_true',
                    help='val mode')


def test(test_dir, model_dir, save_results=False, mode="val"):
    predictor = yolox_predictor.YoloxPredictor()
    if not predictor.loadModel(model_dir):
        logging.error("load model fail")
    else:
        img_path = os.path.join(test_dir, mode)
        T = 0
        current_time = time.localtime()
        outputs = []
        if os.path.isdir(img_path):
            files = predictor.get_image_list(img_path)
            files.sort()
            for img_id, img_name in enumerate(files):
                t0 = time.time()
                img = cv2.imdecode(np.fromfile(img_name, dtype=np.uint8), cv2.IMREAD_COLOR)
                results = predictor.smartPredict(img)
                if save_results:
                    save_folder = os.path.join(model_dir, "vis_res")
                    if not os.path.exists:
                        os.makedirs(save_folder, exist_ok=True)
                    predictor.visual(img_name, save_folder, results, current_time)
                predictor.save_results(img_id, results, outputs)
                t1 = time.time()
                T += (t1 - t0)
                logging.info("{}/{}, Infer time: {:.4f}ms".format((img_id + 1), len(files), (t1 - t0) * 1000))
            
            result_filename = os.path.join(model_dir, "test_results10.json")
            save_path1 = os.path.join(model_dir, "eval10.json")
            save_path2 = os.path.join(model_dir, "eval20.json")
            origin_filename = os.path.join(test_dir, "annotations/{}_annotations.json".format(mode))

            with open(result_filename, 'w') as jf:
                json.dump(outputs, jf, indent=4)

            if os.path.exists(origin_filename):
                # result = getAPForRes(result_filename, origin_filename, "bbox")
                # print("result:", result)
                results1 = detseg_eval.eval(origin_filename, result_filename, 0.4, "obj")
                with open(save_path1, 'w', encoding='utf-8') as ef:
                    json.dump(results1, ef, indent=4)
                    
                results2 = detseg_eval2.eval(origin_filename, result_filename, 0.4, "obj")
                with open(save_path2, 'w', encoding='utf-8') as ef:
                    json.dump(results2, ef, indent=4)

            logging.info("mTime: {}".format(T / len(files) * 1000))
        else:
            logging.error("image path {} is not dir !".format(img_path))


if __name__ == "__main__":
    args = parser.parse_args()
    test_dir = args.test_dir
    model_dir = args.model_dir
    save_results = args.save_result
    is_val = args.val

    print("save_results:", save_results)

    # predictor = yolox_predictor.createInstance(config_path=os.path.join(model_dir, "predictor.json"))
    if is_val:
        test(test_dir, model_dir, save_results, "val")
    else:
        test(test_dir, model_dir, save_results, "test")
