import csv
import argparse
import os
import cv2
from task.classification.mulnet import mulnet_predictor
parser=argparse.ArgumentParser()
parser.add_argument('-m','--model_dir', default='shared_dir/train_result_dir/mulnet_model', # 需要
                             help='model_dir, dir of model to be loaded')
parser.add_argument('-t','--test_dir', default='shared_dir/train_dir/mulnet_dataset', # 需要
                             help='test_dir, dir of dataset be tested')

if __name__=="__main__":
    args=parser.parse_args()
    model_dir=args.model_dir
    test_dir=args.test_dir
    predictor= mulnet_predictor.createInstance()
    predictor.loadModel(model_dir)
    # predictor.exportONNX()
    for dirpath,dirnames,filenames  in os.walk(test_dir):
        for img_name in filenames:
            img_path=os.path.join(dirpath,img_name)
            # img=cv2.imread(img_path)
            # result=predictor.predict(img_path,info={"part_id":""})
            result=predictor.predict(img_path,info={"part_id": img_path.split("#")[-1]})
            print(result)
    # predictor.exportOnnx()
    # if not os.path.exists(test_dir+"/test_result/"):
    #     os.makedirs(test_dir+"/test_result/")
    # output_csv=test_dir+"/test_result/result.csv"
    # test_label=test_dir+"/label.txt"
    # with open(output_csv,'w',newline="") as f_write:
    #     result_writer=csv.writer(f_write)
    #     with open(test_label,'r',newline="", encoding='utf-8') as f:
    #         label_reader=csv.reader(f,delimiter = ',')
    #         count=0
    #         for row in label_reader:
    #             if count==0:
    #                 count+=1
    #                 continue
    #             elif count==2:
    #                 break
    #             print(row)
    #             row = row[0].split()
    #             img_path=row[0]
    #             label=row[1]
    #             img=cv2.imread(img_path)
    #             info={"part_id":img_path.split("#")[-1].split(".")[0]}
    #             result=predictor.predict(img,info=info)
    #             print(result)
