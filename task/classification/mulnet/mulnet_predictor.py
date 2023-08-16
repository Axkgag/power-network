import imp
import os
import pickle5 as pickle
import json
import time
import typing
import cv2
from typing import Dict,Any,List

import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np

from riemann_interface.defaults import AbstractPredictor
from .lib.train.main_for_focal import setupForTrain
from .lib.utils.config import Config
from .lib.base_predictor import Predictor
from .lib.networks_for_focal import SiameseNet
from .lib.aug import maker_border_resize
# import onnxruntime

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

resolution=256


def createInstance(config_path=None):
    predictor = MulnetPredictor()
    return predictor

class MulnetPredictor(AbstractPredictor):

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
        self.model = SiameseNet( resnet_name=config["resnet_name"], 
                                num_classes=config["num_classes"])
        self.cuda=(config["use_cuda"] and torch.cuda.is_available())
        self.device = "cuda:{}".format(config["gpus"])
        return

    def loadModel(self, model_dir_path: str) -> bool:
        predictor_json_path=os.path.join(model_dir_path,"predictor.json")
        weight_path=os.path.join(model_dir_path,"model_param","model-weight.pkl")
        if not os.path.exists(weight_path) or not os.path.exists(predictor_json_path):
            return False
        with open(predictor_json_path,'r') as f:
            predictor_config=json.load(f)
            if predictor_config == self.config:
                return True
        self.setupPredictor(predictor_config)
        if self.cuda:
            state_dict = torch.load(weight_path)
            self.model.load_state_dict(state_dict)
            self.model.cuda()
            # self.model.to(self.device)
        else:
            state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
        self.model_path_=model_dir_path
        
        # print(next(self.model.parameters()).device)
        # self.session=onnxruntime.InferenceSession(self.model_path_+"/mulnet.onnx")
        return True

    def modelPath(self):
        return self.model_path_
    
    def predict(self,img_path):
        return self.stackPredict([img_path])
    
    def predict(self, img: Any, info: Dict[str, Any] = None) -> Dict[str, Any]:
        return  self.stackPredict([img],[info])
    
    def centercrop(img):
        h,w,_=img.shape
        h_begin=(h-224)//2
        w_begin=(w-224)//2
        return img[h_begin:h_begin+224,w_begin:w_begin+224 ,:]
    
    def stackPredict(self, img_list: List[Any], info_list: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        
        t1 =time.perf_counter()
        with torch.no_grad():
            self.model.eval()
            inputs=[]
            part_id_list=[]
            result=[]
            for i in range(len(img_list)):
                img=img_list[i]
                if type(img)==str:
                    img2= Image.open(img)
                    # print("img.format",img2.mode)
                    img2=img2.convert("RGB")
                    part_id=img_list[i].split("#")[-1]
                    part_id=part_id.split('.')[0]
                    part_id_list.append(part_id)
                    
                else:
                    # assume input images are BGR order
                    img2=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img2 = Image.fromarray(img2)
                    # part_id_list.append(info_list[i]["part_id"])
                    part_id_list.append(i)
                img2 = maker_border_resize(np.asarray(img2), resolution)
                # print("img 20",img2[120:140,120,0])
                img2 = Image.fromarray(img2)
                # img2=Image.fromarray(cv2.cvtColor( cv2.imread(img),cv2.COLOR_BGR2RGB) ) 
                # img2 = img2.unsqueeze(0)
                
                img2=self.transform(img2)
                inputs.append(img2)
            inputs = torch.stack(inputs,dim=0)

            if self.cuda:
                inputs = inputs.cuda()
                # inputs = inputs.to(self.device)
                # print(inputs.device)
            
            # onnx_score=self.session.run(None,{"input":inputs.cpu().numpy()})[0]

            # print("onnx score",onnx_score)
            scores = self.model(inputs) # B x C           
            scores = scores.cpu().detach().numpy()
            # print("scores:",scores)
            t2 = time.perf_counter()
            # print('time:%.3fs' % (t2 - t1))
            for i in range(len(scores)):
                pred_cls = int(np.argmax(scores[i]))
                obj_class=self.config["classes_name"][pred_cls]
                result.append({"PartID":part_id_list[i], 
                               "obj_class": obj_class,
                               "score":float(scores[i][pred_cls]),
                               "tot":(t2-t1)})

            return result

    def exportONNX(self):
        import torch.onnx
        import numpy as np
        self.model.eval()
        import torch
        x=torch.randn((1,3,224,224)).cuda()
        # input_embedding=input_embedding.cpu().detach().numpy()
        # print("input embedding shape:",input_embedding.shape)
        
        torch.onnx.export(self.model,
                          x,
                          self.model_path_+"/mulnet.onnx",
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=["input"],
                          output_names=["output"]
                          )