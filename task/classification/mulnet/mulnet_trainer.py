from operator import mod, pos
import os
import json
import shutil
from typing import Any, Dict

from numpy.core.numeric import Infinity
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch


from .lib.train.main_for_focal import setupForTrain, save_model
from .lib.train.trainer_for_focal import train_epoch
from .lib.train.merge_conv_bn import merge_bn_cpu, merge_conv_bn
from .lib.train.test_after_merge import testAfterMerge, testData, get_model
from .lib.utils.config import Config
from .lib.networks_for_focal import SiameseNet
from riemann_interface.defaults import  AbstractTrainer,Status,TrainStatus,AbstractSender


def createInstance(sender):
    return MulnetTrainer(sender)

class TrainController:
    def __init__(self) -> None:
        self.stop_signal=False

class MulnetTrainer(AbstractTrainer):

    def __init__(self, sender: AbstractSender):
        super().__init__(sender)
        self.task_type="classification"


    def setupTrain(self, train_dir: str, is_resume: bool=False) -> Status:
        config_path=train_dir+"/trainer.json"
        cache_dir=train_dir+"/cache"
        self.train_dir = train_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        with open(config_path, encoding='utf-8') as f:
            train_config=json.load(f)
        
        export_dir= train_config["export_dir"]
        if export_dir is None:
            export_dir=train_dir.replace("train_dir","train_result_dir")
        self.export_dir=export_dir
        
        
        actual_config=Config(config_path=config_path)
        for key in train_config:
            setattr(actual_config,key,train_config[key])
        self.config=actual_config
        self.cache_dir=cache_dir
        
        self.cuda=self.config.use_cuda
        self.log_interval=100
        self.metrics=[]
        self.start_epoch=0
        self.end_epoch=self.config.epochs
        self.config.steps = [int(self.config.epochs * step) for step in self.config.steps]
        self.train_loader,self.val_loader, self.model, self.loss_fn,self.ohem_fn,self.optimizer,self.scheduler , self.transform=setupForTrain(self.config)
        self.best=Infinity
        if is_resume:
            self.load_tmp_model()
            self.start_epoch=self.current_epoch+1
            
        status=Status()
        status.success=True
        return status
        
    def save_tmp_model(self):
        this_model_name=os.path.join(self.cache_dir, 'model_stop.pkl')
        torch.save(self.model.state_dict(), this_model_name,_use_new_zipfile_serialization=False)
    
    def load_tmp_model(self):
        state_dict=torch.load(os.path.join(self.cache_dir, 'model_stop.pkl'))
        self.model.load_state_dict(state_dict)

    def trainEpoch(self,epoch:int) -> TrainStatus:
        this_model_name=os.path.join(self.cache_dir, 'model_stop.pkl')
        
        if self.stop_signal == True:
            self.stop_signal=False
            return
        self.current_epoch=epoch
        
        print("----------")
        print("TrainEpoch: ", epoch)

        train_loss, metrics = train_epoch(self.train_loader, 
                                          self.model, 
                                          self.loss_fn, 
                                          self.ohem_fn, 
                                          self.optimizer, 
                                          self.cuda,
                                          self.log_interval, 
                                          self.metrics)

        self.scheduler.step()
        train_status=TrainStatus()
        train_status.train_loss=train_loss
        train_status.epoch=epoch
        print("self.best",self.best)
        if train_loss<self.best :
            best_model_name=os.path.join(self.cache_dir, "model_best.pkl")
            self.best=train_loss
            torch.save(self.model.state_dict(), best_model_name,_use_new_zipfile_serialization=False)
            
        last_model_name=os.path.join(self.cache_dir, 
                                     'model_{}.pkl'.format(self.current_epoch - self.config.save_interval))
        if self.config.save_interval > 0 and self.current_epoch % self.config.save_interval == 0:
            pass
            
        self.last_save_weight=this_model_name
        
        return train_status
        # if train_status_sender :
        #     train_status_sender.send(data={"epoch":epoch,"train_loss":train_loss,"train_accuracy":0,"val_loss":0,"val_accuracy":0},method="train_stat")
        # print(message)
    
    def exportModel(self, export_path: str):
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        weight_dir=export_path+"/model_param"
        report_dir=export_path+"/report"
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        # if os.path.abspath(self.train_dir) != os.path.abspath(export_path):
        #     shutil.copy(self.train_dir+"/trainDataInfo.json",export_path+"/trainDataInfo.json")
        # merge bn
        self.config.model_bn_path=self.cache_dir+"/model_best.pkl"
        self.config.model_nobn_path=weight_dir+"/model-weight.pkl"
        # merge_bn_cpu(weight_dir+"/model-weight.pkl",self.cache_dir+"/model_best.pkl",self.config)
        merge_conv_bn(self.config)
            
        # test 
        self.config.model_nobn_path=weight_dir+"/model-weight.pkl"
        self.config.train_result_path=report_dir+"/train_result.csv"
        
        # with open(export_path+"/predictor.json",'w')as f:
        #     f.write(json.dumps({ "resnet_name": self.config.get("cls_model_name"),"num_classes":self.config.get("num_classes")  }))
        # if os.path.exists(os.path.join(self.train_dir, "trainer.json")):
        #     shutil.copy(os.path.join(self.train_dir, "trainer.json"), os.path.join(export_path, "predictor.json"))
        predictor_obj={"resnet_name":self.config.cls_model_name, 
                       "use_cuda":True,
                       "num_classes":self.config.num_classes,
                       "classes_name":self.config.classes_name}
        with open(os.path.join(export_path, "predictor.json"),'w') as f:
            json.dump(predictor_obj,f,indent=4)
            f.close()
        test_result=testAfterMerge(self.config)
        print("test_result",test_result)
        self.exportONNX(export_path)
        self.sender.send(method="train_result",data={"task_type":self.task_type,"confusion_matrix":test_result.tolist()})#mulnet
        return True
    
    def exportONNX(self,export_path: str):
        import torch.onnx
        import numpy as np
        
        import torch
        x=torch.randn((1,3,224,224)).cuda()
        # input_embedding=input_embedding.cpu().detach().numpy()
        # print("input embedding shape:",input_embedding.shape)
        model_pkl_path=os.path.join(export_path,"model_param","model-weight.pkl")
        print("model pkl path",model_pkl_path)
        self.model = SiameseNet( resnet_name=self.config.cls_model_name, 
                                num_classes=self.config.num_classes)
        self.model.load_state_dict(torch.load(model_pkl_path))
        self.model.cuda()
        self.model.eval()
        torch.onnx.export(self.model,
                          x,
                          export_path+"/mulnet.onnx",
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=["input"],
                          output_names=["output"]
                          )  