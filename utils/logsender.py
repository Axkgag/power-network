import requests
import json
import os
from typing import Dict
from riemann_interface.defaults import AbstractSender,TrainStatus

class LogSender(AbstractSender):
    def __init__(self) -> None:
        self.log_dir_path="./log/"
        if not os.path.exists(self.log_dir_path):
            os.mkdir(self.log_dir_path)
        
    def connect(self,new_address=None,new_port=None):
        pass
            
    def send(self, data: Dict[str, int or float or str or Dict] or TrainStatus, method: str = 'msg') -> None:
        if(isinstance(data,TrainStatus)) :
            result={}
            result["train_loss"]=data.train_loss
            result["val_loss"]=data.val_loss
            result["train_accuracy"]=data.train_accuracy
            result["val_accuracy"]=data.val_accuracy
            result["epoch"]=data.epoch
            print("data_train,result",data.train_loss)
            data=result
        if not os.path.exists(self.log_dir_path) :
            os.makedirs(self.log_dir_path)
        with open(os.path.join(self.log_dir_path,"log.txt"),mode='a') as f:
            f.write((json.dumps({"method":method,"data":data})+"\n"))
            f.close()
            
    def setParam(self, param: Dict) -> bool:
        if "log_dir_path" in param :
            self.log_dir_path=param["log_dir_path"]
            with open(os.path.join(self.log_dir_path,"log.txt"),mode='w') as f:
                f.write("")
                f.close()
        return True