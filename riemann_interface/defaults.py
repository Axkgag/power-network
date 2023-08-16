# -*- coding: utf-8 -*-

from abc import abstractmethod
from cmath import inf
from typing import Any, Dict, Union, List, Tuple
import threading
import os
import csv
import time
import torch
import traceback
import numpy as np
from pycocotools.coco import COCO
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import copy
import cv2
import math
import datetime
import json
class TrainStatus:
    def __init__(self):
        self.epoch=0
        self.train_loss=0
        self.train_accuracy=0
        self.val_loss=0
        self.val_accuracy=0

class Status:
    def __init__(self,success:bool=True,error_str:str=""):
        self.success=success
        self.error_str=error_str
        

class AbstractSender:
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def send(self,data:Dict[str,int or float or str or Dict] or TrainStatus,method:str='msg')->None:
        raise NotImplemented

class AbstractPredictor():
    """推理类基类
    需要实现3个接口：
    setupPredictor(self,config:Dict[str,Any])
    loadModel(self,model_dir_path:str)->bool
    predict(self,img:Any, info:Dict[str,V]=None)->Dict[str,Any]
    可选： stackPredict(self,img_list:List[Any], info_list:List[Dict[str,Any]]=None)->List[Dict[str,Any]]
    """
    def __init__(self, config:Dict[str,Any]=None):
        self.config=None
        if config is not None:
            self.setupPredictor(self.config)
    
    @abstractmethod
    def setupPredictor(self,config:Dict[str,Any]):
        """加载参数，进行createmodel 等过程
        """
        raise NotImplementedError
    
    @abstractmethod
    def loadModel(self,model_dir_path:str)->bool:
        """从单个文件夹加载predictor所有需要的文件，如果config 与当前模型有结构差异，需要重建model(可调用setupPredictor)
        :model_dir_path:    文件夹路径，一般使用绝对路径
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self,img:Any, info:Dict[str,Any]=None)->Dict[str,Any]:
        """实际预测，读入图片和信息（可选），返回结果
        :img:       图片或图片路径，在函数内判断
        :info:      信息，如点位信息等，以字典形式传入
        :return:    返回预测结果，以字典形式传出
        """
        raise NotImplementedError

    def stackPredict(self,img_list:List[Any], info_list:List[Dict[str,Any]]=None)->List[Dict[str,Any]]:
        result=[]
        for i in range(len(img_list)):
            img=img_list[i]
            info=None 
            if i<len(info_list):
                info=info_list[i]
            result.append(self.predict(img=img,info=info))
        return result

    def filter_by_iou(self, results):
        raise NotImplementedError

    def filter_by_iou_plus(self, results):
        raise NotImplementedError

class AbstractTrainer:
    """训练类基类
    需要实现3个接口：
    setupTrain(self,train_dir:str,is_resume:bool)->Status
    trainEpoch(self)->TrainStatus
    exportModel(self, dirpath:str)
    """
    def __init__(self,sender:AbstractSender):
        """初始化，建立训练状态、训练线程等
        """
        self.begin_thread_ = threading.Thread(group=None,target=self.actualTrain,name="beginThread")
        self.is_training=False
        self.sender=sender
        self.last_save_weight=None  #有保存权重时训练线程可立马终止
        self.stop_signal=False      #无保存权重时训练
        self.train_dir=None
        self.is_resume=False
        self.is_pause=False
  
    @abstractmethod
    def setupTrain(self,train_dir:str,is_resume:bool)->Status:
        """从训练文件夹初始化所有训练参数，包括建立训练模型，在训练时设定是否从上次训练继续
        :train_dir: 训练文件夹
        :is_resume: 是否从上次训练继续,如果为True 
        """
        raise NotImplemented

    @abstractmethod
    def trainEpoch(self,epoch:int)->TrainStatus:
        """单个epoch训练, 所有参数需要通过self的属性传入
        """
        raise NotImplemented
    
    @abstractmethod
    def save_tmp_model(self):
        raise NotImplemented
    
    @abstractmethod
    def load_tmp_model(self):
        raise NotImplemented

    def actualTrain(self)->None:
        # print('in')
        """在实际训练线程中进行的训练, 所有参数需要通过self的属性传入，其中start_epoch 和 end_epoch 需要在setupTrain中 完成 
        """
        try:
            status=self.setupTrain(self.train_dir,self.is_resume)
        except Exception as ex:
            status=Status(success=False,error_str="setup_fail"+str(traceback.format_exc()))
            print("ex",traceback.format_exc())
        if not self.is_resume:
            self.last_save_weight=None
        if not status.success or not hasattr(self,"start_epoch") or not hasattr(self,"end_epoch"):
            
            self.sender.send(data={"success":status.success,"error_str":str(status.error_str)},method="setup_fail")
            return
        train_status=TrainStatus()
        for epoch in range(self.start_epoch,self.end_epoch):
            print("self.signal",self.stop_signal)
            try:
                train_status=self.trainEpoch(epoch)
                self.sender.send(data=train_status,method="train_status")
            except Exception as ex:
                print("ex",traceback.format_exc())
                self.sender.send(data={"exception":str(traceback.format_exc(limit=1))},method="train_fail")
                return
            if self.stop_signal:
                self.stop_signal=False
                break
                
        self.is_training=False
        
        if not self.is_pause:
            self.sender.send(data=train_status,method="finish_train")
            try:
                self.exportModel(self.export_dir)
                self.sender.send(data={"export_dir":self.export_dir,"task_type":self.task_type},method="finish_export")
            except Exception as ex:
                traceback.print_exc()
                self.sender.send(data={"exception":str(traceback.format_exc())},method="export_fail")
        else:
            print("has save_tmp_model:",hasattr(self,"save_tmp_model"))
            if hasattr(self,"save_tmp_model"):
                self.save_tmp_model()
            self.is_pause=False
                        
    def beginTrain(self,train_dir:str,is_resume:bool=False)->bool:
        """外部调用的接口，开始训练，初始化训练文件夹和训练状态，启动训练线程
        :train_dir:     训练文件夹
        :is_resume:     是否从上次训练继续
        """
        # 分块
        if not os.path.exists(train_dir):
            print("no dir",train_dir)
            return False
        # if self.is_training ==True:
        #     return False
        self.train_dir=train_dir
        self.is_resume=is_resume
        
        # if hasattr(self,"begin_thread_" ) and self.begin_thread_:
        #     del self.begin_thread_
        # self.begin_thread_ = threading.Thread(group=None,target=self.actualTrain,name="beginThread")
        # self.begin_thread_.start()
        self.actualTrain()
        self.is_training=True
        # self.begin_thread_.join()
        return True

    def resumeTrain(self)->bool:
        """继续训练
        """
        if not self.train_dir:
            return False
        return self.beginTrain(self.train_dir,is_resume=True)
        
    def endTrain(self)->bool:
        """结束训练：如果有已保存权重，直接停止线程，否则通过stop signal 在下个epoch 停止
        """
        if self.is_training == False:
            if self.train_dir:
                self.sender.send(data={},method="finish_train")
                try:
                    self.exportModel(self.export_dir)
                    self.sender.send(data={"export_dir":self.export_dir,"task_type":self.task_type},method="finish_export")
                except Exception as ex:
                    traceback.print_exc()
                    self.sender.send(data={"exception":str(traceback.format_exc())},method="export_fail")
        try:
            if hasattr(self,"emptyCache"):
                self.emptyCache()
            for i in range(5):
                torch.cuda.empty_cache()
            print(torch.cuda.memory_summary())
            self.sender.send(data={"mem_sum":str(torch.cuda.memory_summary())},method='mem_sum')
        except Exception as ex:
            print("ex",ex)
            self.sender.send(data={"exception":str(traceback.format_exc())},method="fail_empty_cache")
        self.is_training=False
        # if self.last_save_weight is not None:
        #     del self.begin_thread_
        # else:
        self.stop_signal=True
        # print("self.signal",self.stop_signal)
        return True
    
    def pauseTrain(self)->bool:
        """结束训练：如果有已保存权重，直接停止线程，否则通过stop signal 在下个epoch 停止
        """
        if self.is_training ==False:
            return False
        self.is_training=False
        # if self.last_save_weight is not None:
        #     del self.begin_thread_
        # else:
        self.stop_signal=True
        self.is_pause=True
        # print("self.signal",self.stop_signal)
        return True
        
    @abstractmethod
    def exportModel(self, export_path:str)->bool:
        """导出所有需要的文件到 export_path,确保在 predictor load model 时可用
        """
        raise NotImplemented


class AbstractObjectDetectTrainer(AbstractTrainer):
    """目标检测训练类基类
    需要实现3个接口：
    setupTrain(self,train_dir:str,is_resume:bool)->Status
    trainEpoch(self)->TrainStatus
    exportModel(self, dirpath:str)
    """

    def __init__(self, sender: AbstractSender):
        """初始化，建立训练状态、训练线程等
        """
        self.begin_thread_ = threading.Thread(group=None, target=self.actualTrain, name="beginThread")
        self.is_training = False
        self.sender = sender
        self.last_save_weight = None  # 有保存权重时训练线程可立马终止
        self.stop_signal = False  # 无保存权重时训练
        self.train_dir = None
        self.is_resume = False
        self.is_pause = False

    @abstractmethod
    def setupTrain(self, train_dir: str, is_resume: bool) -> Status:
        """从训练文件夹初始化所有训练参数，包括建立训练模型，在训练时设定是否从上次训练继续，保证从config中读取 滑窗训练相关参数
        :train_dir: 训练文件夹
        :is_resume: 是否从上次训练继续,如果为True
        """
        raise NotImplemented

    @abstractmethod
    def trainEpoch(self, epoch: int) -> TrainStatus:
        """单个epoch训练, 所有参数需要通过self的属性传入，若启用了滑窗检测训练模式，要做对应的修改
        """
        raise NotImplemented

    def cvtColor(self, image: Any):
        """将图像转换成RGB图像，防止灰度图在预测时报错
        """
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image
        else:
            image = image.convert('RGB')
            return image

    def imageConversion(self, line: List[any], train_img_dir: str, resize_shape: Tuple[int, int]):
        """单张图片box坐标映射到resize的图像上
        :line: 一行信息，img_name+box+label
        :return: image_path, image_data, box:  返回图片路径, 1024x1024的ndarray数据, 相对应映射之后的坐标
        """
        image_path = line[0]
        image_path = os.path.join(train_img_dir, image_path)
        image = Image.open(image_path)
        image = self.cvtColor(image)
        iw, ih = image.size
        w, h = resize_shape

        box = np.array(line[1:])
        new_image = image.resize((w, h), Image.BICUBIC)

        if len(box) > 0:
            box[:, [0, 2]] = box[:, [0, 2]] * w / iw
            box[:, [1, 3]] = box[:, [1, 3]] * h / ih
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        image_data = np.array(new_image)
        return image_path, image_data, box

    def setupDataset(self, train_annotation_path: str, train_img_dir: str, crop_image_dir: str, crop_json_path: str):
        """对COCO格式数据集中的图片进行crop处理，产生新的数据和标签，并保存到默认位置（模型训练时使用新产生的数据集训练）。
           输入参数自定义。
        :train_annotation_path:  crop之前coco数据的.json文件路径
        :train_img_dir:          crop之前数据图片的路径
        :crop_image_dir:         crop后图片存放的文件夹，提前定义好
        :crop_json_path:         crop之后生成coco数据的.json文件路径，提前定义好
        """

        # crop_x, crop_y = self.crop_w, self.crop_h
        # stride_x, stride_y = self.stride_x, self.stride_y

        # img_w, img_h = img.size()
        # w_r, h_r = int((img_w - crop_w) / x_stride + 1), int((img_h - crop_h) / y_stride + 1)
        #
        # size = 640
        # h_r, w_r = 3, 3
        # stride = 320

        # 原始图片根据滑动次数，需要先resize到一定的大小
        # resize_shape = (size + (w_r-1) * stride, size + (h_r-1) * stride)

        if not os.path.exists(crop_image_dir):
            os.makedirs(crop_image_dir)

        annDir = train_annotation_path
        with open(annDir, encoding='utf-8') as f:
            ann_info = json.load(f)
            if len(ann_info["annotations"][0]["bbox"]) == 5:
                mode = 'Rotation_Detection'
            else:
                mode = 'Detection'


        # step 1 读取数据，信息保存在all_img_info里面
        coco = COCO(annotation_file=train_annotation_path)
        ids = list(sorted(coco.imgs.keys()))
        coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])

        all_img_info = []
        for img_id in ids[0: len(ids)]:
            img_info = []
            ann_ids = coco.getAnnIds(imgIds=img_id)
            targets = coco.loadAnns(ann_ids)
            height = coco.loadImgs(img_id)[0]['height']
            width = coco.loadImgs(img_id)[0]['width']
            path = coco.loadImgs(img_id)[0]['file_name']
            img_info.append(path)
            img_info.append(width)
            img_info.append(height)

            if mode == 'Detection':
                for target in targets:
                    x, y, w, h = target["bbox"]
                    x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
                    label = target["category_id"]
                    bbox = [x1, y1, x2, y2, label]
                    img_info.append(bbox)
                all_img_info.append(img_info)
            else:
                for target in targets:
                    x, y, w, h, angle = target["bbox"]
                    x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
                    label = target["category_id"]
                    bbox = [x1, y1, x2, y2, angle, label]
                    img_info.append(bbox)
                all_img_info.append(img_info)

        # step2 数据集中的图片进行crop处理，产生新的数据和标签，相关信息保存在new_all_img_info里面
        new_all_img_info = []
        for line in all_img_info:
            img_w = line.pop(1)
            img_h = line.pop(1)

            if self.crop_w and self.crop_h and self.stride_x and self.stride_y:
                crop_x_list = [self.crop_w]
                crop_y_list = [self.crop_h]
                stride_x_list = [self.stride_x]
                stride_y_list = [self.stride_y]
            else:
                crop_x_list = [img_w, 800]
                crop_y_list = [img_h, 800]
                stride_x_list = [0.8 * x for x in crop_x_list]
                stride_y_list = [0.8 * y for y in crop_y_list]

            for crop_x, crop_y, stride_x, stride_y in zip(crop_x_list, crop_y_list, stride_x_list, stride_y_list):
                crop_x = int(crop_x)
                crop_y = int(crop_y)
                stride_x = int(stride_x)
                stride_y = int(stride_y)

                w_r, h_r = int((img_w - crop_x) / stride_x + 1), int((img_h - crop_y) / stride_y + 1)
                resize_shape = (crop_x + (w_r - 1) * stride_x, crop_y + (h_r - 1) * stride_y)

                image_path, image_data, box = self.imageConversion(line, train_img_dir, resize_shape)
                (file_path, img_name) = os.path.split(image_path)
                box_t1 = box

                for i in range(h_r):
                    for j in range(w_r):
                        new_img_info = []
                        box_t = copy.deepcopy(box_t1)
                        img_crop = image_data[i * stride_y: i * stride_y + crop_y,
                                   j * stride_x: j * stride_x + crop_x]
                        if len(box_t) == 0:
                            # file = img_name[:-4] + "_" + "{}".format(i) + "_" + "{}".format(j) + ".jpg"
                            # new_image_path = os.path.join(crop_image_dir, file)
                            # img_crop = img_crop[..., ::-1]
                            # # cv2.imwrite(new_image_path, img_crop)
                            # cv2.imencode('.jpg', img_crop)[1].tofile(new_image_path)
                            # new_img_info.append(file)

                            # 原图中没有检测框？应该直接删除
                            continue
                        else:
                            # 先算映射的box，以该滑窗区域左上角顶点为坐标原点
                            box_t[:, [0, 2]] = box_t[:, [0, 2]] - j * stride_x
                            box_t[:, [1, 3]] = box_t[:, [1, 3]] - i * stride_y

                            index = []  # 需要删除的框的id
                            for k in range(len(box_t)):
                                if mode == 'Detection':
                                    if box_t[k][0] < 0 or box_t[k][2] > crop_x or box_t[k][1] < 0 or box_t[k][3] > crop_y:
                                        index.append(k)
                                else:
                                    x1, y1, x2, y2, angle, label = box_t[k]
                                    result = [int((x2 + x1) / 2), int((y1 + y2) / 2), int(x2 - x1), int(y2 - y1), angle]
                                    result = np.array(result)
                                    x = int(result[0])
                                    y = int(result[1])
                                    height = int(result[2])
                                    width = int(result[3])
                                    anglePi = (result[4] + 90) / 180 * math.pi
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

                                    x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
                                    y0n = (x0 - x) * sinA + (y0 - y) * cosA + y

                                    x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
                                    y1n = (x1 - x) * sinA + (y1 - y) * cosA + y

                                    x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
                                    y2n = (x2 - x) * sinA + (y2 - y) * cosA + y

                                    x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
                                    y3n = (x3 - x) * sinA + (y3 - y) * cosA + y

                                    if x0n < 0 or x0n > crop_x or x1n < 0 or x1n > crop_x or \
                                            x2n < 0 or x2n > crop_x or x3n < 0 or x3n > crop_x or \
                                            y0n < 0 or y0n > crop_y or y1n < 0 or y1n > crop_y or \
                                            y2n < 0 or y2n > crop_y or y3n < 0 or y3n > crop_y:
                                        index.append(k)
                            index = tuple(index)
                            # 删掉边界box
                            new_box_t = np.delete(box_t, index, axis=0)
                            if len(new_box_t) != 0:
                                file = img_name[:-4] + "_" + "{}".format(crop_x) + "_" + "{}".format(i) + "_" + "{}".format(j) + ".jpg"
                                new_image_path = os.path.join(crop_image_dir, file)
                                # 保存图片
                                img_crop = img_crop[..., ::-1]
                                # cv2.imwrite(new_image_path, img_crop)
                                cv2.imencode('.jpg', img_crop)[1].tofile(new_image_path)
                                new_img_info.append(file)
                                new_img_info.append(crop_x)
                                new_img_info.append(crop_y)

                                if mode == 'Detection':
                                    for n in range(len(new_box_t)):
                                        x1, y1, x2, y2 = new_box_t[n][0], new_box_t[n][1], new_box_t[n][2], new_box_t[n][3]
                                        label = new_box_t[n][4]
                                        bbox = [x1, y1, x2, y2, label]
                                        new_img_info.append(bbox)
                                else:
                                    for n in range(len(new_box_t)):
                                        x1, y1, x2, y2 = new_box_t[n][0], new_box_t[n][1], new_box_t[n][2], new_box_t[n][3]
                                        label = new_box_t[n][4]
                                        angle = new_box_t[n][5]
                                        bbox = [x1, y1, x2, y2, label, angle]
                                        new_img_info.append(bbox)

                                new_all_img_info.append(new_img_info)

        # step3 将信息写入到指定路径的json中
        coco_classes = coco_classes
        annotations: List[Dict[str, Union[int, List[int], List[Any]]]] = []
        categories: List[Dict[str, Union[int, str]]] = []
        images: List[Dict[str, Union[str, int]]] = []

        # 图片
        image_id: int = 0
        annotation_id: int = 1
        for img_info in new_all_img_info:
            file_name = img_info[0]
            image = {
                "date_captured": "",
                "file_name": file_name,
                "height": int(img_info[2]),
                "id": image_id,
                "license": 1,
                "width": int(img_info[1])
            }
            images.append(image)

            # 标注
            for bbox in img_info[3:]:
                if mode == 'Detection':
                    x1, y1, x2, y2, label = bbox
                    w = x2 - x1
                    h = y2 - y1
                    annotation = {
                        "area": int(w * h),
                        "bbox": [
                            int(x1),
                            int(y1),
                            int(w),
                            int(h)
                        ],
                        "category_id": int(label),
                        "id": annotation_id,
                        "image_id": image_id,
                        "iscrowd": 0,
                        "segmentation": []
                    }
                else:
                    x1, y1, x2, y2, angle, label = bbox
                    w = x2 - x1
                    h = y2 - y1
                    annotation = {
                        "area": int(w * h),
                        "bbox": [
                            int(x1),
                            int(y1),
                            int(w),
                            int(h),
                            int(angle)
                        ],
                        "category_id": int(label),
                        "id": annotation_id,
                        "image_id": image_id,
                        "iscrowd": 0,
                        "segmentation": []
                    }
                annotations.append(annotation)
                annotation_id += 1
            image_id += 1
        save_time = datetime.datetime.now()

        # 标签
        for id, name, in coco_classes.items():
            category = {
                "id": id,
                "name": name
            }
            categories.append(category)
        # 汇总
        annotations_dict = {
            'annotations': annotations,
            'categories': categories,
            'images': images,
            'info': {
                "contributor": "Riemann",
                "date_created": save_time.strftime("%Y-%m-%d_%H:%M:%S"),
                "description": "Exported from RIVIP-Learn",
                "url": "",
                "version": "1",
                "year": "2022"
            }
        }
        open(crop_json_path, 'w+', encoding="utf-8").write(
            json.dumps(annotations_dict, sort_keys=True, indent=4, separators=(',', ': ')))


@abstractmethod
def resultToStrList(result:Dict)->List[str]:
    raise NotImplemented
    
    
def testPredictor(predictor:AbstractPredictor, data_folder:str,test_label:str,other_info_role:Dict[str,int],output_csv="result.csv"):
    print(predictor.loadModel(data_folder))
    with open(output_csv,'w',newline="") as f_write:
        result_writer=csv.writer(f_write)
        with open(test_label,'r',newline="", encoding='utf-8') as f:
            label_reader=csv.reader(f,delimiter = ',')
            count=0
            for row in label_reader:
                if count==0:
                    count+=1
                    continue
                print(row)
                row = row[0].split()
                img_path=row[0]
                label=row[1]
                # other_info={}
                # for role in other_info_role:
                #     other_info_role[role]=row[other_info_role[role]]
                result=predictor.predict(img_path)
                print(result)
                # result_writer.writerow( row.extend(resultToStrList(result)))

# def testPredictor(predictor:AbstractPredictor, data_folder:str,test_label:str,other_info_role:Dict[str,int],output_csv="result.csv"):
#     print(predictor.loadModel(data_folder))
#     predictor.testFolder()
#     # result = predictor.predict('D:/Projects/异常检测/mvtec_anomaly_detection/carpet/test/color/000.png')
#     # print(result)


def testTrainer(trainer:AbstractTrainer,train_dir:str,wait_time:int):
    trainer.beginTrain(train_dir)
    time.sleep(wait_time)
    trainer.endTrain()
    time.sleep(wait_time)
    trainer.resumeTrain()
    time.sleep(wait_time)
    trainer.endTrain()

class AbstractObjectDetectPredictor(AbstractPredictor):
    """目标检测推理基类
    """

    def setCropConfig(self, config: Dict) -> bool:
        key_list = ["crop_w", "crop_h", "stride_x", "stride_y"]
        for key in key_list:
            if (not key in config) or not isinstance(config[key], int):
                return False
        self.crop_w = config["crop_w"]
        self.crop_h = config["crop_h"]
        self.stride_x = config["stride_x"]
        self.stride_y = config["stride_y"]
        return True

    @abstractmethod
    def setupPredictor(self, config: Dict[str, Any]):
        """加载参数，进行createmodel 等过程
        """
        self.use_patch = self.setCropConfig(config)
        return True

    # def patchPredict(self, img: Any, info: Dict[str, Any] = None):
    #     """patchPredict 滑窗检测
    #
    #     Args:
    #         img (Any): opencv image
    #         info (Dict[str,Any], optional): 信息. Defaults to None.
    #
    #     Returns:
    #         result (Dict): 与正常predict一样形式的检测结果
    #     """
    #     result_list = []
    #     for crop_img in crops:
    #         result_list.append(crop_start + self.predict(crop_img))
    #     return postprocess(result_list)

    def patchPredict(self, img: os.PathLike, info: Dict[str, Any] = None):
        """patchPredict 滑窗检测

       Args:
           img (Any): 图片文件路径
           info (Dict[str,Any], optional): 信息. Defaults to None.

       Returns:
           result (Dict): 与正常predict一样形式的检测结果
       """
        pass

    def patchPredict(self, img: Any, info: Dict[str, Any] = None):
        """输入一张图像img，实现分块预测，并将检测结果合并。
        :img:          图片或图片路径，在函数内判断
        :mode:         检测任务模式， 'Detection' or 'Rotation_Detection', 默认为'Rotation_Detection'
        :slide_num:    滑动次数，默认为3，3x3
        """
        # crop_x, crop_y = self.crop_w, self.crop_h
        # stride_x, stride_y = self.stride_x, self.stride_y

        h, w, c = img.shape
        # h_r = w_r = 3  # 滑动次数
        # stride = 256  # 滑动步长, 实际滑动步长为512-stride

        if self.crop_w and self.crop_h and self.stride_x and self.stride_y:
            crop_x_list = [self.crop_w]
            crop_y_list = [self.crop_h]
            stride_x_list = [self.stride_x]
            stride_y_list = [self.stride_y]
        else:
            crop_x_list = [w, 800]
            crop_y_list = [h, 800]
            stride_x_list = [0.8 * x for x in crop_x_list]
            stride_y_list = [0.8 * y for y in crop_y_list]

        new_results = {}  # 所有bbox的预测结果

        for crop_x, crop_y, stride_x, stride_y in zip(crop_x_list, crop_y_list, stride_x_list, stride_y_list):
            crop_x = int(crop_x)
            crop_y = int(crop_y)
            stride_x = int(stride_x)
            stride_y = int(stride_y)

            w_r, h_r = int((w - crop_x) / stride_x + 1), int((h - crop_y) / stride_x + 1)
            W, H = crop_x + stride_x * (w_r - 1), crop_y + stride_y * (h_r - 1)
            image_resize = cv2.resize(img, dsize=(W, H))

            # 汇总box
            for i in range(h_r):
                for j in range(w_r):
                    img_crop = image_resize[i * stride_y: i * stride_y + crop_y,
                               j * stride_x: j * stride_x + crop_x]
                    # 一个滑窗预测的结果
                    ret_crop = self.predict(img_crop)
                    ret = ret_crop["results"]
                    if len(ret) == 0:
                        continue
                    # cls_id, bbox_scores分别是预测类别和预测结果
                    # cls_id.size=1, bbox_scores.size=5(包含x1,y1,x2,y2,angle,probability)
                    for cls_id, bbox_scores in ret.items():
                        for n in range(len(bbox_scores)):
                            bbox = bbox_scores[n][0:4]
                            if len(bbox_scores[n]) == 6:  # if mode == 'Rotation_Detection':
                                ang = bbox_scores[n][4]
                                score = bbox_scores[n][5]
                            else:
                                score = bbox_scores[n][4]
                            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                            # 映射到切割前的图片上（1024*1024）
                            x1 = max(0, np.floor(x1).astype('int32')) + np.floor(j * stride_x).astype('int32')
                            y1 = max(0, np.floor(y1).astype('int32')) + np.floor(i * stride_y).astype('int32')
                            x2 = min(img_crop.shape[1], np.floor(x2).astype('int32')) + np.floor(
                                j * stride_x).astype(
                                'int32')
                            y2 = min(img_crop.shape[0], np.floor(y2).astype('int32')) + np.floor(
                                i * stride_y).astype(
                                'int32')
                            h_scale = h / H
                            w_scale = w / W
                            # 映射到resize前的图像上（原始尺寸）
                            x1 = np.floor(x1 * w_scale).astype('int32')
                            y1 = np.floor(y1 * h_scale).astype('int32')
                            x2 = np.floor(x2 * w_scale).astype('int32')
                            y2 = np.floor(y2 * h_scale).astype('int32')
                            if len(bbox_scores[n]) == 6:
                                bbox_score = [x1, y1, x2, y2, ang, score]
                            else:
                                bbox_score = [x1, y1, x2, y2, score]

                            if cls_id not in new_results.keys():
                                new_results[cls_id] = []
                            new_results[cls_id].append(bbox_score)

        # 进行nms过滤
        results = self.filter_by_iou_plus(new_results)
        # results = self.filter_by_iou(new_results)

        # 过滤掉置信度小于0.3的框
        final_results = {}
        for cat_id in results:
            cat_result = results[cat_id]
            filtered_cat_result = []

            for bbox_score in cat_result:
                if len(bbox_score) == 6:
                    score = bbox_score[5]
                else:
                    score = bbox_score[4]
                if score < 0.3:
                    continue

                filtered_cat_result.append(bbox_score)
            final_results[cat_id] = filtered_cat_result

        ret = {"results": final_results}
        return ret

    def smartPredict(self, img: Any, info: Dict[str, Any] = None):
        if self.use_patch:
            return self.patchPredict(img, info)
        else:
            return self.predict(img, info)
