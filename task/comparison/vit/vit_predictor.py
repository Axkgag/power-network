from genericpath import exists
import os
import pickle5 as pickle
import json
import time
import cv2

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from typing import Dict,Any,List
from riemann_interface.defaults import AbstractPredictor
from task.comparison.clsnet.lib.utils.config import Config

from .lib.base_predictor import Predictor
from .lib.networks_for_focal import SiameseNet
from .lib.aug import maker_border_resize
from flask import current_app

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

resolution=256


def createInstance(config_path=None):
    #config_path=os.path.join(os.path.dirname(__file__),"configs","predictor_configs",config_name+".json")
    predictor = ClsnetPredictor()

    return predictor

class ClsnetPredictor(AbstractPredictor):

    def __init__(self, config: Dict[str, Any] = None):
        self.transform=transforms.Compose([
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        self.model_path_=""
        super().__init__(config)

    def setupPredictor(self, config: Dict[str, Any]):
        self.config=config
        if "resnet_name" in  self.config :
            self.config["cls_model_name"]=self.config["resnet_name"]
        self.model = SiameseNet(self.config["cls_model_name"])
        self.cuda= config["use_cuda"] and torch.cuda.is_available()

    def load_obj(self,name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def loadModel(self, model_dir_path: str) -> bool:
        self.part_threshold_={}
        # self.part_resize_={}
        with open( os.path.join(model_dir_path,"model_param","threshold.json")) as f:
            self.part_threshold_=json.load(f)
            f.close()
        # with open(os.path.join(model_dir_path,"model_param","part-resize.json")) as f:
        #     self.part_resize_=json.load(f)
        #     f.close()
        embedding_path=os.path.join(model_dir_path,"standard_data","std_embeddings.pkl")
        predictor_config_path=model_dir_path+"/predictor.json"
        weight_path=os.path.join(model_dir_path,"model_param","model-weight.pkl")
        if not os.path.exists(predictor_config_path) or   not os.path.exists(embedding_path) or not os.path.exists(weight_path):
            return False
        with open(predictor_config_path,'r',encoding='utf-8') as f:
            preditor_config=json.load(f)
            # if preditor_config == self.config:
            #     return True
            self.setupPredictor(preditor_config)
            
        if "resnet_name" in  self.config :
            self.config["cls_model_name"]=self.config["resnet_name"]
        if  "gpus" in self.config:
            os.environ['CUDA_VISIBLE_DEVICES'] =  self.config["gpus"]
        self.model = SiameseNet(self.config["cls_model_name"])
        self.cuda= self.config["use_cuda"] and torch.cuda.is_available()
        # current_app.logger.debug("loadmodel, use_cuda:{}".format(str(self.cuda)))
        torch.no_grad()
        self.model.eval()
        
        self.std_embeddings = self.load_obj(embedding_path)
        self.model_path_=model_dir_path
        if self.cuda:
            for part_id in self.std_embeddings:
                print("part,len",part_id,len(self.std_embeddings[part_id]))
                for t in range(len(self.std_embeddings[part_id])):
                    self.std_embeddings[part_id][t]=self.std_embeddings[part_id][t].cuda()
            state_dict = torch.load(weight_path)
            self.model.load_state_dict(state_dict)
            self.model.cuda()
        else:
            state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
            
        return True
    
    def modelPath(self):
        return self.model_path_
    
    
    def predict(self, img: Any, info: Dict[str, Any] = None) -> Dict[str, Any]:
        return super().predict(img, info)

    
    def stackPredict(self, img_list: List[Any], info_list: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        t1 =time.perf_counter()
        compare_num=5
        with torch.no_grad():
            input_embedding=[]
            part_id_list=[]
            result=[]
            for i in range(len(img_list)):
                img=img_list[i]
                
                if type(img)==str:
                    
                    img2= Image.open(img) 
                    if img2.mode != 'RGB':
                        img2=img2.convert('RGB')
                    part_id=img_list[i].split("#")[-1].split('.')[0]
                    part_id_list.append(part_id)
                else:
                    # assume input images are BGR order
                    img2=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img2 = Image.fromarray(img2)
                    part_id=info_list[i]["part_id"]
                    part_id_list.append(part_id)
                    
                    
                if part_id in self.config["not_keep_ratio_part"]:
                    img2 = img2.resize((resolution, resolution))
                else:
                    img2 = maker_border_resize(np.asarray(img2), resolution)
                    img2 = Image.fromarray(img2)
                img2 = self.transform(img2)
                # img2 = img2.unsqueeze(0)
                input_embedding.append(img2)
            input_embedding=torch.stack(input_embedding,dim=0)
            if self.cuda:
                input_embedding=input_embedding.cuda()
            #     current_app.logger.debug("use gpu: True")
            # else:
            #     current_app.logger.debug("use gpu: False")
            input_data=self.model.get_embedding(input_embedding)
            thrs=[]
            for part_id in part_id_list :
                if part_id in self.part_threshold_:
                    thrs.append(self.part_threshold_[part_id])
                else:
                    thrs.append(0.2)
            thrs = np.array(thrs, dtype=np.float32)
            predictions=[]
            scores=np.zeros((1,len(part_id_list)),np.float)
            for t in range(compare_num):
                # print("t",t)
                std_data=[]
                for part_id in part_id_list:
                    t=min(t,len(self.std_embeddings[part_id])-1)
                    if t<0:
                        print("error t for part", part_id, t, len(self.std_embeddings[part_id]))
                        std_data.append(self.std_embeddings[part_id][0])
                    else:
                        std_data.append(self.std_embeddings[part_id][t])
                
                std_data = torch.stack(std_data, dim=0)
                std_data = std_data.squeeze(1)
                outputs = self.model.get_fc(std_data,input_data) # 21 x 2
                outputs = torch.nn.Softmax(dim=1)(outputs)
                binary_scores = outputs.cpu().detach().numpy()
                binary_scores=binary_scores[:, 1] 
                prediction = binary_scores > thrs
                predictions.append(prediction)
                scores+=binary_scores
            # print("input shape:",input_data.cpu().detach().numpy().shape)
            scores/=compare_num
            # print("scores:",scores)
            predictions = np.array(predictions, dtype=np.int32)
            predictions = np.sum(predictions, axis=0)
            scores= np.sum(scores, axis=0)
            is_ok =( predictions >= compare_num/2)
            t2 = time.perf_counter()
            print('time:%.3fs' % (t2 - t1))
            for i in range(len(is_ok)):
                result.append({"PartID":part_id_list[i],"result":1-int(is_ok[i]),"score":scores[i],"tot":(t2-t1)})
            return result        
    

