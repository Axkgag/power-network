import socket
import json
from typing import Dict
from riemann_interface.defaults import AbstractSender,TrainStatus

class Sender(AbstractSender):
    def __init__(self,address,port) -> None:
        self.address=address
        self.port=port
        
        self.data_end="#*#DataEnd#*#"
    
    def connect(self,new_address=None,new_port=None):
        if new_address :
            self.address=new_address
            self.port=new_port
            self.socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            b=self.socket.connect((self.address,self.port))
            print(b)
        
    def send(self, data: Dict[str, int or float or str or Dict] or TrainStatus, method: str) -> None:
        try :
            self.socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            b=self.socket.connect((self.address,self.port))
            #print("connect result",b)
            if(isinstance(data,TrainStatus)) :
                result={}
                result["train_loss"]=data.train_loss
                result["val_loss"]=data.val_loss
                result["train_accuracy"]=data.train_accuracy
                result["val_accuracy"]=data.val_accuracy
                result["epoch"]=data.epoch
                print("data_train,result",data.train_loss)
                data=result
            self.socket.sendall(json.dumps({"method":method,"data":data}).encode("utf-8")+self.data_end.encode("utf-8"))
            #print("send success",self.socket.recv(1024).decode("utf8"))
            # print("sending:",method,data)
            self.socket.close()
        except Exception as e:
            print("send fail",e)
        return 

            
            
if __name__=="__main__":
    sender=Sender("192.168.1.86",3005)
    sender.send({"sendtest":"b"},"test")
