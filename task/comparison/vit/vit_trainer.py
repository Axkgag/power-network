import imp
from operator import mod, pos
import os
import json
import shutil
from typing import Any, Dict

from numpy.core.numeric import Infinity
import torch


from .lib.train.main_for_focal import setupForTrain, save_model
from .lib.train.trainer_for_focal import train_epoch
from .lib.train.merge_conv_bn import merge_bn_cpu, merge_conv_bn
from .lib.train.test_after_merge import get_anchor_embeddings, testAfterMerge,testData,save_obj,get_model
from.lib.utils.config import Config
from riemann_interface.defaults import  AbstractTrainer,Status,TrainStatus,AbstractSender


def createInstance(sender):
    return ClsnetTrainer(sender)

class TrainController:
    def __init__(self) -> None:
        self.stop_signal=False

class ClsnetTrainer(AbstractTrainer):

    def __init__(self, sender: AbstractSender):
        super().__init__(sender)
        self.task_type="comparison"


    def setupTrain(self, train_dir: str, is_resume: bool=False) -> Status:
        config_path=train_dir+"/trainer.json"
        cache_dir=train_dir+"/cache"
        self.train_dir = train_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        with open(config_path,'r',encoding='utf-8') as f:
            train_config=json.load(f)
        
        export_dir= train_config["export_dir"]
        if export_dir is None:
            export_dir=train_dir.replace("train_dir","train_result_dir")
        self.export_dir=export_dir
        
        
        actual_config=Config(config_path)
        for key in train_config:
            setattr(actual_config,key,train_config[key])
        self.config=actual_config
        self.cache_dir=cache_dir
        self.config.train_dir = self.train_dir
        
        self.cuda=self.config.use_cuda,
        self.log_interval=1, 
        self.metrics=[]
        self.start_epoch=0
        self.end_epoch=self.config.epochs+1
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
        if self.stop_signal == True:
            self.stop_signal=False
            return
        self.current_epoch=epoch
        
        train_loss, metrics = train_epoch(self.train_loader, 
                                          self.model, 
                                          self.loss_fn, 
                                          self.ohem_fn, 
                                          self.optimizer, 
                                          self.cuda,
                                          1, 
                                          self.metrics)

        self.scheduler.step()
        train_status=TrainStatus()
        train_status.train_loss=train_loss
        train_status.epoch=epoch
        if train_loss<self.best :
            best_model_name=os.path.join(self.cache_dir, "model_best.pkl")
            self.best=train_loss
            torch.save(self.model.state_dict(), best_model_name,_use_new_zipfile_serialization=False)
        print(self.current_epoch)
        print(self.config.save_interval)
        last_model_name=os.path.join(self.cache_dir, 
                                     'model_{}.pkl'.format(self.current_epoch - self.config.save_interval))
        this_model_name=os.path.join(self.cache_dir, 'model_{}.pkl'.format(self.current_epoch))
        if True:#self.config.save_interval > 0 and self.current_epoch % self.config.save_interval == 0:
            if os.path.exists(last_model_name) :
                os.remove(last_model_name)
            torch.save(self.model.state_dict(), this_model_name,_use_new_zipfile_serialization=False)
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
        embd_dir=export_path+"/standard_data"
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        if not os.path.exists(embd_dir):
            os.makedirs(embd_dir)
        #shutil.copy(self.train_dir+"/trainDataInfo.json",export_path+"/trainDataInfo.json")
        # merge bn
        self.config.model_bn_path=self.cache_dir+"/model_best.pkl"
        self.config.model_nobn_path=weight_dir+"/model-weight.pkl"
        # merge_bn_cpu(weight_dir+"/model-weight.pkl",self.cache_dir+"/model_best.pkl",self.config)
        merge_conv_bn(self.config)

        pos_ids=[]
        thres={}
        # part_resize=self.config.not_keep_ratio_part
        for pos_id in self.config.pos_map:
           pos_ids.append(pos_id) 
           thres[pos_id]=0.2
        with open(weight_dir+"/threshold.json",'w') as f:
            json.dump(thres,f)
        # with open(weight_dir+"/part-resize.json",'w') as f:
        #     json.dump(part_resize,f)
        # with open(export_path+"/predictor.json",'w') as f:
        #     json.dump({"use_cuda":True,"resnet_name":self.config.cls_model_name},f)
        if os.path.exists(os.path.join(self.train_dir, "trainer.json")):
            shutil.copy(os.path.join(self.train_dir, "trainer.json"), os.path.join(export_path, "predictor.json"))

        # self.exportOnnx(model_path=self.config.model_bn_path, export_path=export_path)
           
        # test 
        self.config.model_nobn_path=weight_dir+"/model-weight.pkl"
        self.config.std_embd_path=embd_dir+"/std_embeddings"
        self.config.train_result_path=report_dir+"/train_result.csv"
        test_result=testAfterMerge(self.config)
        print("test_result",test_result)
        test_result["task_type"]=self.task_type
        self.sender.send(method="train_result",data=test_result)
        return True
    
    def exportOnnx(self, model_path, export_path):
        import torch.onnx
        from .lib.networks_for_focal import SiameseNet
        x = torch.randn((2, 1, 3, 224, 224)).cuda()
        model = SiameseNet(self.config.cls_model_name)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint)
        model = model.cuda()
        model.eval()

        torch.onnx.export(model,
                          x,
                          os.path.join(export_path, "model.onnx"),
                          export_params=True,
                          opset_version=12,
                          do_constant_folding=True,
                          input_names=["input"],
                          output_names=["output"]
                          )
    
    
