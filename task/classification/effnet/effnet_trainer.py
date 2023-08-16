from operator import mod, pos
import os
import json
import shutil
from typing import Any, Dict
import random
import numpy as np

from numpy.core.numeric import Infinity
import torch

from .lib.train.test_after_train import testAfterTrain
from .lib.train.main_for_focal import setupForTrain
from .lib.train.trainer_for_focal import train_epoch, val
from .lib.utils.config import Config
from riemann_interface.defaults import AbstractTrainer, Status, TrainStatus, AbstractSender

os.environ['PYTHONHASHSEED'] = str(666)  # 为了禁止hash随机化，使得实验可复现
random.seed(666)
np.random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.cuda.manual_seed_all(666)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def createInstance(sender):
    return EffnetTrainer(sender)


class TrainController:
    def __init__(self) -> None:
        self.stop_signal = False


class EffnetTrainer(AbstractTrainer):

    def __init__(self, sender: AbstractSender):
        super().__init__(sender)
        self.task_type = "classification"

    def setupTrain(self, train_dir: str, is_resume: bool = False) -> Status:

        config_path = train_dir + "/trainer.json"
        cache_dir = train_dir + "/cache"
        self.train_dir = train_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        with open(config_path, encoding='utf-8') as f:
            train_config = json.load(f)

        export_dir = train_config["export_dir"]
        if export_dir is None:
            export_dir = train_dir.replace("train_dir", "train_result_dir")
        self.export_dir = export_dir

        actual_config = Config(config_path=config_path)
        for key in train_config:
            setattr(actual_config, key, train_config[key])
        self.config = actual_config
        self.cache_dir = cache_dir

        self.cuda = self.config.use_cuda
        self.log_interval = self.config.log_interval
        self.metrics = []
        self.start_epoch = 0
        self.config.trainDir = self.train_dir
        self.config.val_interval = -1

        self.config.train_file = os.path.join(self.train_dir, "label.txt")
        self.end_epoch = self.config.epochs + 1
        self.train_loader, self.val_loader, self.model, self.criterion, self.optimizer, self.scheduler, self.transform = setupForTrain(
            self.config)
        self.best = Infinity
        if is_resume:
            self.load_tmp_model()
            self.start_epoch = self.current_epoch + 1

        status = Status()
        status.success = True
        return status

    def save_tmp_model(self):
        this_model_name = os.path.join(self.cache_dir, 'model_stop.pkl')
        torch.save(self.model.state_dict(), this_model_name, _use_new_zipfile_serialization=False)

    def load_tmp_model(self):
        state_dict = torch.load(os.path.join(self.cache_dir, 'model_stop.pkl'))
        self.model.load_state_dict(state_dict)

    def trainEpoch(self, epoch: int) -> TrainStatus:
        print("trainEpoch:")
        this_model_name = os.path.join(self.cache_dir, 'model_stop.pkl')

        if self.stop_signal == True:
            self.stop_signal = False
            return
        self.current_epoch = epoch

        train_loss, metrics = train_epoch(self.train_loader,
                                          self.model,
                                          self.criterion,
                                          self.optimizer,
                                          self.cuda,
                                          self.log_interval,
                                          self.metrics)

        self.scheduler.step()
        train_status = TrainStatus()
        train_status.train_loss = train_loss
        train_status.epoch = epoch
        print("self.best", self.best)
        # 是否使用验证集
        if self.config.val_interval > 0:
            if self.current_epoch % self.config.val_interval == 0:
                val_loss = val(self.val_loader, self.model, self.criterion, self.cuda, self.config.log_interval,
                               self.metrics)
                if val_loss < self.best:
                    best_model_name = os.path.join(self.cache_dir, "model_best.pkl")
                    self.best = val_loss
                    torch.save(self.model.state_dict(), best_model_name, _use_new_zipfile_serialization=False)

        elif train_loss < self.best:
            best_model_name = os.path.join(self.cache_dir, "model_best.pkl")
            self.best = train_loss
            torch.save(self.model.state_dict(), best_model_name, _use_new_zipfile_serialization=False)

        self.last_save_weight = this_model_name

        return train_status

    def exportModel(self, export_path: str):
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        #weight_dir = export_path + "/model_param"
        weight_dir = export_path
        report_dir = export_path + "/report"
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        if os.path.abspath(self.train_dir) != os.path.abspath(export_path):
            shutil.copy(self.train_dir + "/trainDataInfo.json", export_path + "/trainDataInfo.json")

        if os.path.exists(weight_dir):
            # shutil.move(os.path.join(self.cache_dir, "model_best.pkl"), os.path.join(weight_dir, "model-weight.pkl"))
            shutil.copy(os.path.join(self.cache_dir, "model_best.pkl"), os.path.join(weight_dir, "model-weight.pkl"))
        # test
        self.config.model_nobn_path = weight_dir + "/model-weight.pkl"
        self.config.train_result_path = report_dir + "/train_result.csv"

        predictor_obj = {"arch": self.config.arch,
                         "use_cuda": True,
                         "num_classes": self.config.num_classes,
                         "classes_name": self.config.classes_name}
        with open(os.path.join(export_path, "predictor.json"), 'w') as f:
            json.dump(predictor_obj, f)
            f.close()
        test_result = testAfterTrain(self.config)
        print("test_result", test_result)
        self.exportONNX(export_path)
        self.sender.send(method="train_result",
                         data={"task_type": self.task_type, "confusion_matrix": test_result.tolist()})  # mulnet
        return True

    def exportONNX(self, export_path: str):
        import torch.onnx
        import numpy as np

        import torch
        import torch.nn as nn
        # from .lib.train.torchvision.models import get_model
        from .lib.networks_for_focal import EfficientNet
        x = torch.randn((1, 3, 224, 224)).cuda()
        model_pkl_path = os.path.join(export_path, "model-weight.pkl")
        model = EfficientNet(self.config.arch, self.config.num_classes)
        self.model.load_state_dict(torch.load(model_pkl_path))
        self.model.cuda()
        self.model.eval()
        torch.onnx.export(self.model,
                          x,
                          export_path + "/model.onnx",
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=["input"],
                          output_names=["output"]
                          )
