import json
import importlib
import os 
import sys
from typing import Any, Dict, List

from riemann_interface.defaults import AbstractPredictor
sys.path.append(os.getcwd())
class PredictorManager():
    def __init__(self):
        self.predictor_dict_:Dict[str,AbstractPredictor]={}
        # scan available task algorithms and configs
        self.available_config_dict_={}
        for root,dirs,files in os.walk(os.path.join(os.getcwd(),"../task")):
            for task_dir in dirs:
                for algorithm_root,algorithm_dirs,alg_files in os.walk(os.path.join(root,task_dir)):
                    for algorithm_dir in algorithm_dirs :
                        for config_root, config_dirs,config_files in os.walk(os.path.join(algorithm_root,algorithm_dir)):
                            self.available_config_dict_[algorithm_dir]=config_files.remove(".json")

    
    def createPredictor(self,task_type:str,algorithm_name:str,config_path:str)->str:
        dir_name=""
        path_split=config_path.split("/")
        for i in range(len(path_split)):
            word=path_split[i]
            if word=="model_data" :
                dir_name=path_split[i+1]
        predictor_name=algorithm_name+"@"+dir_name
        if self.isPredictorExist(predictor_name):
            del self.predictor_dict_[predictor_name]
        predictor_path="task.{1}.{0}.{0}_predictor".format(algorithm_name,task_type)
        # if not os.path.exists(predictor_path):
        #     print("predicter cannot be loaded, file not found",predictor_path)
        #     return False
        predictor_module=importlib.import_module(predictor_path)
        if not hasattr(predictor_module,"createInstance") :
            print("create predictor fail, createInstance not found!")
            return None
        predictor=predictor_module.createInstance(config_path)
        predictor.config_name=config_path
        predictor.task_type=task_type
        predictor.algorithm_name=algorithm_name
        self.predictor_dict_[predictor_name]=(predictor)
        return predictor_name
    
    def getPredictorInfo(self,predictor_name:str)->Dict:
        if self.isPredictorExist(predictor_name):
            info={}
            predictor=self.predictor_dict_[predictor_name]
            info["config_name"]=predictor.config_name
            info["task_type"]=predictor.task_type
            info["algorithm_name"]=predictor.algorithm_name
            return info
        else:
            return None

    def removePredictor(self,predictor_name:str)->None:
        if self.isPredictorExist(predictor_name):
            del self.predictor_dict_[predictor_name]

    def loadModel(self,predictor_name:str,model_dir_path:str)->str:
        success=False
        if self.isPredictorExist(predictor_name):
            success=self.predictor_dict_[predictor_name].loadModel(model_dir_path)
        if success:
            return model_dir_path
        else:
            return None

    def runPredictor(self,predictor_name:str,img_list:list,stack_predict=False,info_list:List[Dict[str,Any]]=None)->Any:
        if self.isPredictorExist(predictor_name) :
            if not stack_predict:
                result=[]
                for i in range(len(img_list)):
                    image=img_list[i]
                    if i <len(info_list):
                        info=info_list[i]
                    result.append(self.predictor_dict_[predictor_name].predict(img=image,info=info))
                return result
            else:
                return self.predictor_dict_[predictor_name].stackPredict(img_list= img_list,info_list=info_list)
        else:
            return None

    def isPredictorExist(self,predictor_name:str)->bool:
        if predictor_name in self.predictor_dict_:
            return True
        else:
            print("algorithm not find",predictor_name)
            return False

    def predictorList(self)->list:
        return list(self.predictor_dict_.keys())
    
    def testFolder(self,predictor_name:str,test_folder:str)->bool:
        if self.isPredictorExist(predictor_name):
            self.predictor_dict_[predictor_name].testFolder(test_folder)
            return True
        else:
            return False

        