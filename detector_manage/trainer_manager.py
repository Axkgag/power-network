import json
import importlib
import os 
import sys
from sys import getsizeof
from typing import Any, Callable, Dict
from task.comparison.clsnet.lib.utils import config
import time
from task.object_detection.centernet.lib.utils.config import Config
from riemann_interface.defaults import AbstractTrainer
sys.path.append(os.getcwd())




class TrainerManager():

    def __init__(self,sender:Any):
        self.BEGINTRAIN=0
        self.ENDTRAIN=1
        self.PAUSETRAIN=2
        self.RESUMETRAIN=3
        
        self.trainer_dict_:Dict[str,AbstractTrainer]={}
        self.sender=sender
        # scan available task algorithms and configs
        self.available_config_dict_={}
        for root,dirs,files in os.walk(os.path.join(os.getcwd(),"../task")):
            for task_dir in dirs:
                for algorithm_root,algorithm_dirs,alg_files in os.walk(os.path.join(root,task_dir)):
                    for algorithm_dir in algorithm_dirs :
                        for config_root, config_dirs,config_files in os.walk(os.path.join(algorithm_root,algorithm_dir)):
                            self.available_config_dict_[algorithm_dir]=config_files.remove(".json")
    
    def createTrainer(self,task_type:str,algorithm_name:str)->str:
        trainer_name=algorithm_name
        if self.isTrainerExist(trainer_name):
            print(trainer_name,"already_exist,will not create")
            return None
        trainer_path="task.{1}.{0}.{0}_trainer".format(algorithm_name,task_type)
        # if not os.path.exists(trainer_path):
        #     print("predicter cannot be loaded, file not found",trainer_path)
        #     return False
        trainer_module=importlib.import_module(trainer_path)
        if not hasattr(trainer_module,"createInstance") :
            print("create trainer fail, createInstance not found!")
            return None
        trainer=trainer_module.createInstance(self.sender)
        trainer.trainer_name=trainer_name
        trainer.task_type=task_type
        trainer.algorithm_name=algorithm_name
        self.trainer_dict_[trainer_name]=(trainer)
        return trainer_name
    
    def getTrainerInfo(self,trainer_name:str)->Dict:
        if self.isTrainerExist(trainer_name):
            info={}
            trainer=self.trainer_dict_[trainer_name]
            info["task_type"]=trainer.task_type
            info["algorithm_name"]=trainer.algorithm_name
            return info
        else:
            return None

    def removeTrainer(self,trainer_name:str)->bool:
        if self.isTrainerExist(trainer_name):
            del self.trainer_dict_[trainer_name]
            return True
        else:
            return False


    def trainerList(self)->list:
        return list(self.trainer_dict_.keys())
        
    def showMemory(self,unit='KB', threshold=1):
        '''查看变量占用内存情况
        :param unit: 显示的单位，可为`B`,`KB`,`MB`,`GB`
        :param threshold: 仅显示内存数值大于等于threshold的变量
        '''
        scale = {'B': 1, 'KB': 1024, 'MB': 1048576, 'GB': 1073741824}[unit]
        for trainer_name in self.trainer_dict_.keys():
            memory = getsizeof(self.trainer_dict_[trainer_name]) // scale
            print(trainer_name, memory)
            

    def isTrainerExist(self,trainer_name:str)->bool:
        if trainer_name in self.trainer_dict_:
            return True
        else:
            print("algorithm not find",trainer_name)
            return False

    def execTrainerOp(self,trainer_name:str,op:int,param:Dict[str,str])->bool:
        if self.isTrainerExist(trainer_name) :
            trainer=self.trainer_dict_[trainer_name]  
            if op==self.BEGINTRAIN:
                return trainer.beginTrain(param["train_dir"])
            elif op==self.ENDTRAIN :
                return trainer.endTrain()
            elif op==self.PAUSETRAIN:
                return trainer.pauseTrain()
            elif op==self.RESUMETRAIN:
                return trainer.resumeTrain()
            
        else:
            return False
        
    def getTrainStats(self,trainer_name:str)->bool:
        if self.isTrainerExist(trainer_name) :
            return self.trainer_dict_[trainer_name].train_stats
        else:
            return False



        