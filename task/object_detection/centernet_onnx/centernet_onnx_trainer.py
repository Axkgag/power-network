from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json

import threading
import time
import os
import shutil

import torch
import torch.utils.data

from .lib.utils.config import Config
from .lib.opts import opts
from .lib.models.model import create_model, load_model, save_model
from .lib.models.data_parallel import DataParallel
from .lib.logger import Logger
from .lib.datasets.dataset_factory import get_dataset
from .lib.trains.train_factory import train_factory


from riemann_interface.defaults import Status, TrainStatus, AbstractSender, AbstractObjectDetectTrainer
from utils.riemann_coco_eval import actualGetAP
def createInstance(sender):
    trainer=CenterNetTrainer(sender)
    return trainer


class CenterNetTrainer(AbstractObjectDetectTrainer):
    
    def __init__(self, sender: AbstractSender):
        super().__init__(sender)
    
    def setupInfo(self, train_dir: str):
        trainer_config_path=os.path.join(train_dir,"trainer.json")
        with open(trainer_config_path,'r',encoding='utf-8') as f:
            train_config=json.load(f)
            annotation_path = os.path.join(train_dir, "annotations.json")
            image_path = os.path.join(train_dir, "image")
            if train_config["patch_train"] == 1:
                self.crop_w = train_config["crop_w"]
                self.crop_h = train_config["crop_h"]
                self.stride_x = train_config["stride_x"]
                self.stride_y = train_config["stride_y"]
                crop_image_dir = os.path.join(train_dir, "crop_image")
                crop_json_path = os.path.join(train_dir, "crop_annotation.json")
                self.setupDataset(annotation_path, image_path, crop_image_dir, crop_json_path)
                train_annot_path = crop_json_path
                img_dir = crop_image_dir
            else:
                train_annot_path = annotation_path
                img_dir = image_path
            train_config["train_annot_path"] = train_annot_path
            train_config["val_annot_path"] = train_config["train_annot_path"]
            train_config["img_dir"] = img_dir
            train_config["train_img_dir"] = train_config["img_dir"]
            train_config["val_img_dir"] = train_config["img_dir"]
            train_config["out_dir"] = train_dir+"-output"
            train_config["train_dir"] = train_dir
            if train_config["export_dir"] is None:
                train_config["export_dir"] = train_dir.replace("train_dir","train_result_dir")
        if not os.path.exists(train_config["export_dir"]) :
            os.makedirs(train_config["export_dir"])
        self.export_dir=train_config["export_dir"]
        self.train_dir = train_dir
        cache_dir=train_dir+"/cache/"
        train_config["cache_dir"]=cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.config=train_config
        opt = opts()
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
        self.Dataset = get_dataset(opt.dataset, opt.task)
        self.opt = opts().update_dataset_info_and_set_heads(opt, self.config)
        self.opt.lr_step = [int(i * self.opt.num_epochs) for i in self.opt.lr_step] # 原始lr_step为值，现为相对总epochs的比例
        self.logger = Logger(self.opt)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpus_str
        self.opt.device = torch.device('cuda' if self.opt.gpus[0] >= 0 else 'cpu')
        self.current_model = ""
        self.val_best = 1e10
        self.train_best = 1e10
        self.stop_signal = False
        self.is_training=True
        self.current_epoch = 0

        self.class_names = self.opt.class_names
        self.task_type = self.config["task"]

    def setupTrain(self, train_dir: str, is_resume: bool=False) -> Status:
        print("train dir",train_dir)
        self.setupInfo(train_dir)
        print('Creating model...')
        model = create_model(self.opt.arch, self.opt.heads, self.opt.head_conv)
        
        optimizer = torch.optim.Adam(model.parameters(), self.opt.lr)
        start_epoch = 0
        
        
            
        Trainer = train_factory[self.opt.task]
        trainer = Trainer(self.opt, model, optimizer)
        trainer.set_device(self.opt.gpus, self.opt.chunk_sizes, self.opt.device)
        
        print('Setting up data...')

        train_loader = torch.utils.data.DataLoader(
            self.Dataset(self.opt, 'train'), 
            batch_size=self.opt.batch_size, 
            shuffle=True,
            num_workers=self.opt.num_workers,
            pin_memory=False,
            drop_last=False
        )
        val_loader = None
        if self.opt.val_annot_path != "":
            val_loader = torch.utils.data.DataLoader(
                self.Dataset(self.opt, 'val'),
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
        
        self.model=model
        self.optimizer=optimizer
        self.start_epoch=start_epoch
        self.end_epoch=self.opt.num_epochs
        self.trainer=trainer
        self.train_loader=train_loader
        self.val_loader=val_loader
        if is_resume:
            self.load_tmp_model()
        status= Status(success=True,error_str="")
        print("finish setup train start at",self.start_epoch)
        return status
  
    def save_tmp_model(self):
        this_model_name=os.path.join(self.opt.cache_dir, 'model_stop.pth')
        print("has save model",this_model_name)
        save_model(this_model_name, 
                        self.current_epoch, self.model, self.optimizer)
    def load_tmp_model(self):
        this_model_name=os.path.join(self.opt.cache_dir, 'model_stop.pth')
        self.model, self.optimizer, self.start_epoch = load_model(
        self.model, this_model_name, self.optimizer, True, self.opt.lr, self.opt.lr_step)
        self.start_epoch +=1

    def trainEpoch(self,epoch:int) -> TrainStatus:
        if self.stop_signal == True:
            return
        self.current_epoch=epoch
        train_status=TrainStatus()
        log_dict_train, _ = self.trainer.train(self.current_epoch, self.train_loader, self.stop_signal)
        torch.cuda.empty_cache()
        train_status.train_loss=log_dict_train["loss"]
        print(self.current_epoch, log_dict_train)
        train_status.epoch=self.current_epoch
        for k, v in log_dict_train.items():
            self.logger.scalar_summary('train_{}'.format(k), v, self.current_epoch)
            self.logger.write('{} {:8f} | '.format(k, v))
        last_model_name=os.path.join(self.opt.cache_dir, 
                                     'model_{}.pth'.format(self.current_epoch - self.opt.val_intervals))
        this_model_name=os.path.join(self.opt.cache_dir, 'model_{}.pth'.format(self.current_epoch))
        if self.opt.val_intervals > 0 and self.current_epoch % self.opt.val_intervals == 0:
            if os.path.exists(last_model_name) :
                os.remove(last_model_name)
            save_model(this_model_name, 
                            self.current_epoch, self.model, self.optimizer)
            self.opt.load_model=this_model_name
            with torch.no_grad():
                log_dict_val, preds = self.trainer.val(self.current_epoch, self.val_loader)
                train_status.val_loss=log_dict_val["loss"]
                if log_dict_val[self.opt.metric] < self.val_best:  
                    self.val_best = log_dict_val[self.opt.metric]
                    save_model(os.path.join(self.opt.cache_dir, 'model_val_best.pth'), 
                            self.current_epoch, self.model)
        
        if self.current_epoch in self.opt.lr_step:
                lr = self.opt.lr * (0.1 ** (self.opt.lr_step.index(self.current_epoch) + 1))
                print('Drop LR to', lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        if  log_dict_train["loss"]<self.train_best or self.current_epoch>=self.end_epoch-1:  
            save_model(os.path.join(self.opt.cache_dir, 'model_train_best.pth'), 
                    self.current_epoch, self.model)
        #print("train status ,loss",train_status.train_loss)
        return train_status
    
    def testTrainDataset(self, save_dir):
        import pycocotools.coco as coco
        from pycocotools.cocoeval import COCOeval
        print("enter test traindaset")
        train_loader = torch.utils.data.DataLoader(
            self.Dataset(self.opt, 'val'), 
            batch_size=1, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        coco_anno = coco.COCO(self.config["train_annot_path"])
        class_name = ['__background__', ] + self.opt.class_names
        _valid_ids = [i for i in range(1, len(class_name)+1)]
        with torch.no_grad():
            _, results = self.trainer.val(0, train_loader)
            print(len(results))
            def _to_float(x):
                return float("{:.2f}".format(x))

            def convert_eval_format(all_bboxes):
                # import pdb; pdb.set_trace()
                detections = []
                for image_id in all_bboxes:
                    for cls_ind in all_bboxes[image_id]:
                        category_id = _valid_ids[cls_ind - 1]
                        for bbox in all_bboxes[image_id][cls_ind]:
                            bbox[2] -= bbox[0]
                            bbox[3] -= bbox[1]
                            score = bbox[4]
                            bbox_out  = list(map(_to_float, bbox[0:4]))

                            detection = {
                                "image_id": int(image_id),
                                "category_id": int(category_id),
                                "bbox": bbox_out,
                                "score": float("{:.2f}".format(score))
                            }
                            if len(bbox) > 5:
                                extreme_points = list(map(_to_float, bbox[5:13]))
                                detection["extreme_points"] = extreme_points
                            detections.append(detection)
                return detections

            def save_results(results, save_dir):
                json.dump(convert_eval_format(results), 
                            open('{}/train_results.json'.format(save_dir), 'w'))
            
            save_results(results, save_dir)
            coco_dets = coco_anno.loadRes('{}/train_results.json'.format(save_dir))
   
            stats=actualGetAP(gt_anno= coco_anno, dets_anno=coco_dets, iou_type= "bbox")
            return stats

  
    def exportModel(self, export_path: str):
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        save_dir=export_path+"/report"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        val_best_model_path=os.path.join(self.opt.cache_dir, 'model_val_best.pth')
        train_best_model_path=os.path.join(self.opt.cache_dir, 'model_train_best.pth')
        if os.path.exists(val_best_model_path):
            shutil.copy(val_best_model_path, os.path.join(export_path,'model_best.pth'))
        else:
            shutil.copy(train_best_model_path, os.path.join(export_path,'model_best.pth'))
        stats=self.testTrainDataset( save_dir)
        self.exportOnnx(export_path,export_path)
        if os.path.exists(os.path.join(self.train_dir, "trainer.json")):
            with open(os.path.join(export_path, "predictor.json"), "w+") as write_f:
                tr_json = json.load(open(os.path.join(self.train_dir, "trainer.json"), "r"))
                pred_json = dict()
                pred_json["arch"] = tr_json["arch"]
                pred_json["score_threshold"] = tr_json["score_threshold"]
                pred_json["iou_thr"] = tr_json["iou_thr"]
                pred_json["class_names"] = tr_json["class_names"]
                pred_json["num_classes"] = tr_json["num_classes"]
                pred_json["input_res"] = tr_json["input_res"]
                pred_json["mean"] = tr_json["mean"]
                pred_json["std"] = tr_json["std"]
                pred_json["gpus"] = tr_json["gpus"]
                pred_json["fp16"] = False
                pred_json["K"]=100
                if tr_json["patch_train"]:
                    pred_json["crop_w"] = self.config["crop_w"]
                    pred_json["crop_h"] = self.config["crop_h"]
                    pred_json["stride_x"] = self.config["stride_x"]
                    pred_json["stride_y"] = self.config["stride_y"]
                # pred_json["patch_predict"] = tr_json["patch_train"]
                # pred_json["crop_size"] = tr_json["crop_size"]
                # pred_json["crop_stride"] = tr_json["crop_stride"]
                json.dump(pred_json, write_f, indent=4, ensure_ascii=False)
            # shutil.copy(os.path.join(self.train_dir, "trainer.json"), os.path.join(export_path, "predictor.json"))
        stats["task_type"]="object_detection"
        print(stats)
        self.sender.send(method="train_result",data=stats)
        return True


    def exportOnnx(self, model_path, export_path):
        import torch.onnx
        x = torch.randn((1, 3, 512, 512)).cuda()
        model = create_model(self.opt.arch, self.opt.heads, self.opt.head_conv)
        model = load_model(model, os.path.join(model_path, "model_best.pth"))
        model = model.cuda()
        model.eval()

        torch.onnx.export(model,
                          x,
                          os.path.join(export_path, "model.onnx"),
                          export_params=True,
                          opset_version=12,
                          do_constant_folding=True,
                          input_names=["input"],
                          output_names=["hm", "wh", "reg"]
                          )


    def verify_onnx_model(self, model_path, onnx_model_path):
        import onnxruntime as rt
        import numpy as np
        sess = rt.InferenceSession(os.path.join(onnx_model_path, "model.onnx"))
        model = create_model(self.opt.arch, self.opt.heads, self.opt.head_conv)
        model = load_model(model, os.path.join(model_path, "model_best.pth"))
        model.eval()

        diff = 0.0
        for i in range(10):
            x = torch.randn((1, 3, 512, 512), dtype=torch.float32)
            model_output = model(x)
            model_output = model_output[-1]
            model_output["hm"] = model_output["hm"].detach().numpy()
            model_output["wh"] = model_output["wh"].detach().numpy()
            model_output["reg"] = model_output["reg"].detach().numpy()


            onnx_input = {sess.get_inputs()[0].name:x.numpy()}
            onnx_output = sess.run(None, onnx_input)

            diff += np.mean(np.abs(model_output["hm"] - onnx_output[0]))
            diff += np.mean(np.abs(model_output["wh"] - onnx_output[1]))
            diff += np.mean(np.abs(model_output["reg"] - onnx_output[2]))
        print("error:", diff/10)


    
    


    
    
  