from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os

import numpy as np
import copy
import cv2
import datetime
from pycocotools.coco import COCO
from PIL import Image
from typing import Any, Dict, Union, List, Tuple

import torch
import torch.utils.data

from .lib.utils.config import Config
from .lib.opts import opts
from .lib.models.model import create_model, load_model, save_model
from .lib.models.data_parallel import DataParallel
from .lib.logger import Logger
from .lib.datasets.dataset_factory import get_dataset
from .lib.trains.train_factory import train_factory
from .lib.utils.voc_eval_utils import voc_eval
import shutil
import math

from riemann_interface.defaults import Status, TrainStatus, AbstractSender, AbstractObjectDetectTrainer

def createInstance(sender):
    trainer=RCenterNetTrainer(sender)
    return trainer

class RCenterNetTrainer(AbstractObjectDetectTrainer):

    def __init__(self, sender: AbstractSender):
        super().__init__(sender)

    def setupInfo(self, train_dir: str):
        trainer_config_path = os.path.join(train_dir, "trainer.json")
        with open(trainer_config_path, 'r', encoding='utf-8') as f:
            train_config = json.load(f)

            keys = ["arch", "batch_size", "class_names", "export_dir", "gpus", "input_res",
                    "lr", "lr_step", "mean", "num_classes", "num_epochs", "num_workers",
                    "score_threshold", "iou_thr", "std", "target_program", "task",
                    "val_intervals", "patch_train", "crop_w", "crop_h", "stride_x", "stride_y"]
            for key in keys:
                if key not in train_config:
                    print("{} is not in config".format(key))
                    return False

            annotation_path = os.path.join(train_dir, "annotations.json")
            image_path = os.path.join(train_dir, "image")
            if train_config["patch_train"] == 1:
                self.crop_w = train_config["crop_w"]
                self.stride_x = train_config["stride_x"]
                self.crop_h = train_config["crop_h"]
                self.stride_y = train_config["stride_y"]
                crop_image_dir = os.path.join(train_dir, "crop_train")
                crop_json_path = os.path.join(train_dir, "crop_annotation.json")
                self.setupDataset(annotation_path, image_path, crop_image_dir, crop_json_path)
                train_annot_path = crop_json_path
                img_dir = crop_image_dir
            else:
                train_annot_path = annotation_path
                img_dir = image_path

            train_config["img_dir"] = img_dir
            train_config["train_dir"] = train_dir
            train_config["train_img_dir"] = train_config["img_dir"]
            train_config["trainval_img_dir"] = train_config["img_dir"]
            train_config["train_annot_path"] = train_annot_path
            train_config["trainval_annot_path"] = train_config["train_annot_path"]

            if ("val_dir" in train_config) and (train_config["val_dir"] != ""):
                train_config["val_annot_path"] = os.path.join(train_config["val_dir"], "annotations.json")
                train_config["val_img_dir"] = os.path.join(train_config["val_dir"], "image")
            else:
                train_config["val_annot_path"] = train_config["train_annot_path"]
                train_config["val_img_dir"] = train_config["img_dir"]

            if ("test_dir" in train_config) and (train_config["test_dir"] != ""):
                train_config["test_annot_path"] = os.path.join(train_config["test_dir"], "annotations.json")
                train_config["test_img_dir"] = os.path.join(train_config["test_dir"], "image")
            else:
                train_config["test_annot_path"] = train_config["train_annot_path"]
                train_config["test_img_dir"] = train_config["img_dir"]

            train_config["out_dir"] = train_dir + "-output"
            if train_config["export_dir"] is None:
                train_config["export_dir"]=train_dir.replace("train_dir","train_result_dir")
        if not os.path.exists(train_config["export_dir"]):
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
        self.class_names_=self.opt.class_names

        self.task_type = "object_detection"

        return True

    def setupTrain(self, train_dir: str, is_resume: bool) -> Status:
        print("train dir",train_dir)
        if not self.setupInfo(train_dir):
            status = Status(success=False, error_str="setupInfo fail")
            return status
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
            # start_epoch=self.current_epoch  +1
        

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
        this_model_name=os.path.join(self.opt.cache_dir, 'model_stop.pth')

        # print("stop sig:",self.stop_signal)
        if self.stop_signal == True:
            return
        self.current_epoch=epoch
        train_status=TrainStatus()
        log_dict_train, _ = self.trainer.train(self.current_epoch, self.train_loader, self.stop_signal)
        torch.cuda.empty_cache()
        train_status.train_loss=log_dict_train["loss"]
        print(self.current_epoch,log_dict_train)
        train_status.epoch=self.current_epoch
        for k, v in log_dict_train.items():
            self.logger.scalar_summary('train_{}'.format(k), v, self.current_epoch)
            self.logger.write('{} {:8f} | '.format(k, v))

        if self.opt.val_intervals > 0 and self.current_epoch % self.opt.val_intervals == 0:
            
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
        
      
    def dump_results(self, coco_anno, save_dir):
        converted_results = {}
        for class_name in self.class_names_:
          converted_results[class_name] = []

        train_results = []

        train_loader = torch.utils.data.DataLoader(
            self.Dataset(self.opt, 'val'), 
            batch_size=1, 
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
        with torch.no_grad():
            _, results = self.trainer.val(0, train_loader)
            img_ids = list(results.keys())
            file_names = [item["file_name"] for item in coco_anno.loadImgs(ids=img_ids)]
            # save into txt format
            for i, file_name in enumerate(file_names):
                det_results = results[img_ids[i]]
                for cat_id in det_results:
                    cat_result = det_results[cat_id]
                    for bbox_score in cat_result:
                        res = copy.deepcopy(bbox_score)
                        res[2] -= res[0]
                        res[3] -= res[1]
                        train_res = {
                            "image_id": int(img_ids[i]),
                            "category_id": cat_id,
                            "bbox": res[:5],
                            "score": res[5]
                        }
                        train_results.append(train_res)

                        score, x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = self.convert_box(bbox_score)
                        converted_results[self.class_names_[cat_id-1]].append(file_name.split(".")[0] + " " + str(score) + " " + str(x0n) + " " + str(y0n) + " " + str(x1n) + " " + str(y1n) 
                                        + " " + str(x2n) + " " + str(y2n) + " " + str(x3n) + " " + str(y3n) + "\n")

        txt_save_dir = os.path.join(save_dir, "txt_train_results")
        if not os.path.exists(txt_save_dir):
            os.mkdir(txt_save_dir)

        for class_name in self.class_names_:
          with open(os.path.join(txt_save_dir, class_name+".txt"), 'w') as f:
            f.writelines(converted_results[class_name])
        with open(os.path.join(save_dir, "train_results.json"), "w") as jf:
            json.dump(train_results, jf)

    def testTrainDataset(self, save_dir, train_mode):
        import pycocotools.coco as coco
        json_path = self.config[train_mode + "_annot_path"]
        # json_path = self.config["train_annot_path"]
        coco_anno = coco.COCO(json_path)
        images_ids = coco_anno.getImgIds()
        file_names = [item["file_name"] for item in coco_anno.loadImgs(ids=images_ids)]
        txt_save_dir = os.path.join(save_dir, "txt_train_results")
        if not os.path.exists(txt_save_dir):
          os.mkdir(txt_save_dir)
        
        gt_save_dir = os.path.join(save_dir, "txt_train_gt")
        if not os.path.exists(gt_save_dir):
          os.mkdir(gt_save_dir)

        self.dump_results(coco_anno, save_dir)

        with open(os.path.join(save_dir, "trainset.txt"), "w") as f:
          for name in file_names:
            f.write(name.split(".")[0] + " " + "\n")
        
        converted_results = {}
        for file_name in file_names:
          file_name = file_name.split(".")[0]
          converted_results[file_name] = []

        for i, image_id in enumerate(images_ids):
          annIds = coco_anno.getAnnIds(imgIds=[image_id])
          anns = coco_anno.loadAnns(annIds)
          file_name = file_names[i].split(".")[0]
          for ann in anns:
            cat_id = self.class_names_[ann["category_id"]-1]
            x, y, w, h, ang = ann["bbox"]
            # bbox_score = [x - w/2., y - h/2., x + w/2., y + h/2., ang] + [1]
            bbox_score = [x, y, x + w, y + h, ang] + [1]
            score, x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = self.convert_box(bbox_score)
            converted_results[file_name].append(str(x0n) + " " + str(y0n) + " " + str(x1n) + " " + str(y1n) 
                                  + " " + str(x2n) + " " + str(y2n) + " " + str(x3n) + " " + str(y3n) +  " " + cat_id + " " + "0" + "\n")

        for file_name in file_names:
          file_name = file_name.split(".")[0]
          with open(os.path.join(gt_save_dir, file_name+".txt"), 'w') as f:
            f.writelines(converted_results[file_name])


        detpath = txt_save_dir + r'/{:s}.txt'
        annopath = gt_save_dir + r'/{:s}.txt'
        imagesetfile = save_dir + r'/trainset.txt'
        classnames = self.class_names_
        classaps = []
        map = 0
        stats = {}
        
        for classname in classnames:
            rec, prec, ap = voc_eval(detpath,
                annopath,
                imagesetfile,
                classname,
                ovthresh=0.5,
                use_07_metric=True)
            map = map + ap
            classaps.append({
                "class_name": classname,
                "ap": [ap, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            })
        map = map / len(classnames)

        stats["map"] = [map, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        stats["class_ap_list"] = classaps
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

        status = []
        datasets = []
        train_status = self.testTrainDataset(save_dir, "train")
        status.append(train_status)
        datasets.append(1)
        if self.config["val_img_dir"] != self.config["train_img_dir"]:
            val_status = self.testTrainDataset(save_dir, "val")
            status.append(val_status)
            datasets.append(2)
        if self.config["test_img_dir"] != self.config["train_img_dir"]:
            test_status = self.testTrainDataset(save_dir, "test")
            status.append(test_status)
            datasets.append(3)

        results = {"stats": status, "datasets": datasets, "task_type": "ObjectDetection"}
        self.sender.send(method="train_result", data=results)
        # stats=self.testTrainDataset( save_dir)
        # stats["task_type"]="object_detection"
        # print(stats)
        # self.sender.send(method="train_result",data=stats)
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
                          output_names=["hm", "wh", "ang", "reg"]
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
            model_output["ang"] = model_output["ang"].detach().numpy()
            model_output["reg"] = model_output["reg"].detach().numpy()


            onnx_input = {sess.get_inputs()[0].name:x.numpy()}
            onnx_output = sess.run(None, onnx_input)

            diff += np.mean(np.abs(model_output["hm"] - onnx_output[0]))
            diff += np.mean(np.abs(model_output["wh"] - onnx_output[1]))
            diff += np.mean(np.abs(model_output["ang"] - onnx_output[2]))
            diff += np.mean(np.abs(model_output["reg"] - onnx_output[3]))
        print("error:", diff/10) 

    def convert_box(self, bbox_score):
        bbox = bbox_score[0:4]
        ang = bbox_score[4]
        score = bbox_score[5]
        result = np.array([int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2),int(bbox[2]-bbox[0]),int(bbox[3]-bbox[1]), ang])
        x, y, width, height = int(result[0]), int(result[1]), int(result[2]), int(result[3])
        anglePi = result[4]/180 * math.pi
        anglePi = anglePi if anglePi <= math.pi else anglePi - math.pi

        cosA = math.cos(anglePi)
        sinA = math.sin(anglePi)
        
        x1=x-0.5*width   
        y1=y-0.5*height
        
        x0=x+0.5*width 
        y0=y1
        
        x2=x1            
        y2=y+0.5*height 
        
        x3=x0   
        y3=y2
        
        x0n= int((x0 -x)*cosA -(y0 - y)*sinA + x)
        y0n = int((x0-x)*sinA + (y0 - y)*cosA + y)
        
        x1n= int((x1 -x)*cosA -(y1 - y)*sinA + x)
        y1n = int((x1-x)*sinA + (y1 - y)*cosA + y)
        
        x2n= int((x2 -x)*cosA -(y2 - y)*sinA + x)
        y2n = int((x2-x)*sinA + (y2 - y)*cosA + y)
        
        x3n= int((x3 -x)*cosA -(y3 - y)*sinA + x)
        y3n = int((x3-x)*sinA + (y3 - y)*cosA + y)
        return score, x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n



