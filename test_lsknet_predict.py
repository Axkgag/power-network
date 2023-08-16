import os
import time
import json
from task.object_detection.lsknet import lsknet_predictor
import argparse
import logging
import cv2
import numpy as np
from utils.riemann_coco_eval import getAPForRes

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir',
                    default='./datasets/lsknet/train/export/',
                    help='model_dir, dir of model to be loaded')
parser.add_argument('-t', '--test_dir',
                    default='./datasets/lsknet/test/',
                    help='test_dir, dir of dataset be tested')
parser.add_argument('-s', '--save_result',
                    default=True,
                    action='store_true',
                    help='save result or not')


if __name__ == "__main__":
    args = parser.parse_args()
    test_dir = args.test_dir
    model_dir = args.model_dir

    predictor = lsknet_predictor.createInstance(config_path=os.path.join(model_dir, "predictor.json"))
    if not predictor.loadModel(model_dir):
        logging.error("load model fail")
    else:
        img_path = os.path.join(test_dir, "image")
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
                if args.save_result:
                    save_folder = os.path.join(test_dir, "vis_res")
                    if not os.path.exists:
                        os.makedirs(save_folder, exist_ok=True)
                    predictor.visual(img_name, save_folder, results, current_time)
                    predictor.save_results(img_id, results, outputs)
                t1 = time.time()
                T += (t1 - t0)
            if args.save_result:
                result_filename = os.path.join(test_dir, "results.json")
                origin_filename = os.path.join(test_dir, "annotations.json")
                with open(result_filename, 'w') as jf:
                    json.dump(outputs, jf, indent=4)
                result = getAPForRes(result_filename, origin_filename, "bbox")
                print("result:", result)
            print("mTime:", T / len(files) * 1000)
        else:
            logging.error("image path {} is not dir !".format(img_path))
    
        # predictor.run_eval(
        #     os.path.join(test_dir, "val"),
        #     os.path.join(test_dir, "val_annotations.json"),
        #     test_dir
        # )
