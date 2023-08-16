import json
import os
import cv2
import numpy
import shutil
import argparse
from typing import List

def coco2stage2(
        origin_dataset_path:os.PathLike,
        stage1_dataset_path:os.PathLike,
        stage2_dataset_path:os.PathLike,
        splitter:str='-',
        stage2_section:int=1
    ):
    """coco2stage2 input a coco dataset, output a stage1 coco dataset,a stage2 label.txt dataset

    Args:
        origin_dataset_path (os.PathLike): _description_
        stage1_dataset_path (os.PathLike): _description_
        stage2_dataset_path (os.PathLike): _description_
    """
    #make dirs and define file paths
    stage1_image_path=os.path.join(stage1_dataset_path,"image")
    stage2_image_path=os.path.join(stage2_dataset_path,"image")
    if os.path.exists(stage1_dataset_path):
        shutil.rmtree(stage1_dataset_path)
    os.makedirs(stage1_dataset_path)
    shutil.copytree(os.path.join(origin_dataset_path,"image"),stage1_image_path)

    if os.path.exists(stage2_dataset_path):
        shutil.rmtree(stage2_dataset_path)
    os.makedirs(stage2_image_path)

    origin_annotation_json_path= os.path.join(origin_dataset_path,"annotations.json")
    stage1_annotation_json_path=os.path.join(stage1_dataset_path,"annotations.json")
    stage2_label_txt=os.path.join(stage2_dataset_path,"label.txt")


    with open(origin_annotation_json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
        cat_list=data["categories"]
        image_list=data["images"]
        annotaion_list=data["annotations"]
        info_list=data["info"]

        #get stage1 stage2 cat lists and str-id dict
        stage1_cat_dict={}
        stage1_cat_list=[]
        stage1_id_str_dict={}
        stage2_cat_dict={}
        stage2_cat_list=[]
        for cat_info in cat_list:
            cat_str_list:List[str]=cat_info["name"].split(splitter)
            cat_str_list=list(filter(lambda x: x != '', cat_str_list))
            if len(cat_str_list)==0 or stage2_section>=len(cat_str_list):
                print("categorty error, could not split to multi section",cat_info["name"],"using splitter",splitter)
                shutil.rmtree(stage1_dataset_path)
                shutil.rmtree(stage2_dataset_path)
                print("finish clean dirs")
                return
            cat_str=cat_info["name"].split(splitter)[0]
            cat2_str=cat_info["name"].split(splitter)[stage2_section]
            if not cat_str in stage1_cat_dict:
                id_num=len(stage1_cat_list)
                stage1_cat_dict[cat_str]=id_num
                stage1_cat_list.append({
                    "id":id_num,
                    "name":cat_str,
                    "supercategory":"none"
                })
            cat_id=cat_info["id"]
            if not cat_id in stage1_id_str_dict:
                stage1_id_str_dict[cat_id]=cat_str
            if not cat2_str in stage2_cat_dict:
                stage2_cat_dict[cat2_str]=len(stage2_cat_list)
                stage2_cat_list.append(cat2_str)

        #conver annotations
        stage1_anno_list=[]
        stage2_anno_list=[]
        total_num=len(annotaion_list)
        print("begin conver annotations,total:",total_num)
        count=0
        for anno in annotaion_list:
            #conver cat id to stage id
            anno_id=anno["id"]
            anno_cat_id=anno["category_id"]
            cat_str=stage1_id_str_dict[anno_cat_id]
            cat_stage1_id=stage1_cat_dict[cat_str]
            anno["category_id"]=cat_stage1_id
            cat2_str=cat_str.split(splitter)[stage2_section]
            stage1_anno_list.append(anno)

            #crop image and save for stage 2
            image_name=image_list[anno["image_id"]]["file_name"]
            image_path=os.path.join(origin_dataset_path,"image",image_name )
            stage2_image=cv2.imdecode(numpy.fromfile(image_path, dtype=numpy.uint8), -1)
            box=anno["bbox"]
            croped_image=stage2_image[box[1]:box[1]+box[3],box[0]:box[0]+box[2],:]
            dot_pos=image_name.rfind(".")
            croped_image_name=image_name[0:dot_pos]+"@"+str(anno_id)+image_name[dot_pos:]
            save_path=os.path.join(stage2_dataset_path,"image",croped_image_name)
            cv2.imencode(".jpg",croped_image)[1].tofile(save_path)
            if not os.path.exists(save_path):
                print("saveing ",save_path,"fail")

            #conver label to stage 2 cat id and add to list
            stage2_anno_list.append(croped_image_name+" "+str(stage2_cat_dict[cat2_str]))
            count+=1
            print("progress: num:{}  percent:{:.2f}%".format(count,count/total_num*100),end='\r')

        data["annotations"]=annotaion_list
        data["categories"]=stage1_cat_list
        with open(stage1_annotation_json_path,'w',encoding='utf8')  as f1:
            json.dump(data,f1)

        with open(stage2_label_txt,'w',encoding='utf8') as f2:
            f2.writelines("\n".join(stage2_anno_list))

parser=argparse.ArgumentParser()
parser.add_argument("-i","--input_dataset_path",help="input coco dataset",required=True)
parser.add_argument("-o1","--output_stage1_path",help="output stage 1 path",required=True)
parser.add_argument("-o2","--output_stage2_path",help="output stage 2 path",required=True)
parser.add_argument("-sp","--splitter",help="label splitter",default='-',type=str)
parser.add_argument("-sec2","--stage2_section",help="label splitter",default=1,type=int)
if __name__=="__main__":
    args=parser.parse_args()
    coco2stage2(
        origin_dataset_path=args.input_dataset_path,
        stage1_dataset_path=args.output_stage1_path,
        stage2_dataset_path=args.output_stage2_path,
        splitter=args.splitter,
        stage2_section=args.stage2_section)