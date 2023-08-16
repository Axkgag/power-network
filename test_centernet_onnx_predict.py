from riemann_interface.defaults import testPredictor
# from task.object_detection.centernet_onnx import centernet_onnx_predictor
from task.object_detection.centernet_onnx import centernet_onnx_predictor
from utils.sender import Sender
import os
import cv2
import json
import numpy as np
from time import time
from PIL import ImageDraw,Image,ImageFont
import matplotlib.pyplot as plt
import argparse
from utils.riemann_coco_eval import getAPForRes

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir',
                    default='../../ChipCounter/data/chip/',
                    help='model_dir, dir of model to be loaded')
parser.add_argument('-t', '--test_dir',
                    default='../../ChipCounter/data/chip/datasets/normal/test/',
                    help='test_dir, dir of dataset be tested')
parser.add_argument('-s', '--save_result',
                    default=False,
                    action='store_true',
                    help='save result or not')

if __name__=="__main__":
    draw_font=ImageFont.truetype("./ttf/Alibaba-PuHuiTi-Regular.ttf", 26, encoding="utf-8")
    color_list = []
    args = parser.parse_args()
    test_dir = args.test_dir
    model_dir = args.model_dir

    if not os.path.exists(test_dir+"image_out/") and args.save_result:     # predict 预测结果输出路径
        os.makedirs(test_dir+"image_out/")

    for i in range(256):
        color_list.append(tuple(((i*17)%256, (i+20)*67%256, (i+60)*101%256)))

    predictor=centernet_onnx_predictor.createInstance(config_path=model_dir+"predictor.json")
    predictor.loadModel(model_dir)

    # -------------------------------------------- #

    # 模型评价指标mAP
    # save_dir = test_dir + "/map"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # predictor.run_eval(test_dir+"crop_val", os.path.join(test_dir, "crop_val_annotations.json"), save_dir)

    # --------------------------------------------- #

    T=0
    cat_index_map={}
    outputs = []
    for root,dirs,files in os.walk(test_dir+"image"):        # img_true：需要预测的数据集的路径
        for img_id, filename in enumerate(files):
            img_path=os.path.join(root,filename)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            t1=time()

            # result=predictor.predict(img)  # 滑动预测
            results=predictor.smartPredict(img)

            # print("predictime ",t2-t1,"\n \t result",result)
            if args.save_result:
                img=Image.open(img_path).convert('RGB')
                cat_index=0
                for cat_id in results["results"]:
                    cat_result=results["results"][cat_id]
                    if not cat_id in cat_index_map:
                        cat_index_map[cat_id]=len(cat_index_map)
                    cat_index=cat_index_map[cat_id]
                    for bbox_score in cat_result:
                        # print("box",bbox_score)
                        bbox = bbox_score[0:4]
                        score = bbox_score[4]
                        # if score < self.score_threshold[key-1]: continue
                        draw=ImageDraw.Draw(img)

                        draw.text( (int(bbox[0]),int(bbox[1])), str(cat_id),fill=color_list[cat_index],font=draw_font)
                        draw.text( (int(bbox[0]),int(bbox[1])-20), str(score),fill=color_list[cat_index],font=draw_font)

                        draw.rectangle(tuple(bbox),outline=color_list[cat_index],width=4)
                # plt.imshow(img)
                # plt.show()
                out_img_path=test_dir+"image_out/"+filename
                # print("out imgname",out_img_path)
                img.save(out_img_path)
                predictor.save_results(img_id, results, outputs)
            t2 = time()
            T += t2 - t1
        if args.save_result:
            result_filename = os.path.join(test_dir, "results.json")
            origin_filename = os.path.join(test_dir, "annotations.json")
            with open(result_filename, 'w') as jf:
                json.dump(outputs, jf, indent=4)
            result = getAPForRes(result_filename, origin_filename, "bbox")
            print("result:", result)
    mT = T / len(files) * 1000
    print("mTime:", mT)
