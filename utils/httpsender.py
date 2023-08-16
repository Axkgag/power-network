import requests
import json
import os
from typing import Dict
from riemann_interface.defaults import AbstractSender,TrainStatus

class HttpSender(AbstractSender):
    def __init__(self) -> None:
        self.address="0.0.0.0"
        self.port=3006
        
    def connect(self,new_address=None,new_port=None):
        self.address=new_address
        self.port=new_port
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
        try :
            requests.post(self.address+":"+str(self.port)+"/"+method,json={"method":method,"data":data})
        except Exception as e:
            print("send fail",e,self.address,self.port)
        return
            
    def setParam(self, param: Dict) -> bool:
        if "log_dir_path" in param :
            self.log_dir_path=param["log_dir_path"]
            with open(os.path.join(self.log_dir_path,"log.txt"),mode='w') as f:
                f.write("")
                f.close()
        return True