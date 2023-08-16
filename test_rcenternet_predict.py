import os
import json
from time import time
from task.object_detection.rcenternet import rcenternet_predictor
import cv2
import argparse
import numpy as np
from utils.riemann_voc_eval import getAPForRes

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir',
                    default='../../ChipCounter/data/chip/test/crop/export/',
                    help='model_dir, dir of model to be loaded')
parser.add_argument('-t', '--test_dir',
                    default='../../ChipCounter/data/chip/test/crop/',
                    help='test_dir, dir of dataset be tested')
parser.add_argument('-s', '--save_result',
                    default=True,
                    action='store_true',
                    help='save result or not')

if __name__ == "__main__":
    args = parser.parse_args()
    test_dir = args.test_dir
    model_dir = args.model_dir
    predictor = rcenternet_predictor.createInstance(config_path=model_dir+"predictor.json")
    if predictor.loadModel(model_dir):

        # 计算Map
        # predictor.run_eval(os.path.join(test_dir, "val"), os.path.join(test_dir, "val_annotations.json"), test_dir)

        # 预测输出的路径

        output_dir = test_dir + "image_out"
        if not os.path.exists(output_dir) and args.save_result:
            os.makedirs(output_dir)

        T = 0
        id = 0

        outputs = []
        for root, dirs, files in os.walk(test_dir+"image"):   # image_test：预测数据文件夹
            for img_id, filename in enumerate(files):
                img_path = os.path.join(root, filename)
                print(img_path)
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
                t1 = time()

                # choose   predict or patchPredict
                # ret = predictor.predict(img)
                ret = predictor.smartPredict(img)    # 滑动预测,3x3


                # 单张可视化预测结果
                # predictor.show_bboxes(img, results=ret["results"])
                # predictor.show_bboxes(img, results=ret["results"])     # 滑动预测可视化

                # 保存预测结果
                # predictor.save_img(img, results=ret["results"], filename=filename, output_dir=output_dir)        # 不滑动预测
                if args.save_result:
                    predictor.save_img(img, results=ret["results"], filename=filename, output_dir=output_dir)          # 滑动预测
                    predictor.save_results(img_id, ret, outputs)
                t2 = time()
                # print("predictime ", t2 - t1, ret)
                t = t2 - t1
                # print(t)
                T += t
            if args.save_result:
                result_filename = os.path.join(test_dir, "results.json")
                origin_filename = os.path.join(test_dir, "annotations.json")
                with open(result_filename, "w") as jf:
                    json.dump(outputs, jf, indent=4)
                result = getAPForRes(result_filename, origin_filename, test_dir)
                print("result:", result)
        # print(T)
        mTime = T / len(files) * 1000
        print("mTime:", mTime)

        # predictor.verify_filter_by_iou_function()
    else:
        print("load model failed")
