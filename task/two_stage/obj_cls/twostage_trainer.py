import json
import os
import threading
import traceback
import torch

from riemann_interface.defaults import AbstractSender, Status, AbstractObjectDetectTrainer, AbstractTrainer, TrainStatus

def createInstance(sender, predictor, classifier):
    trainer = Obj_ClsTrainer(sender, predictor, classifier)
    return trainer

class Obj_ClsTrainer(AbstractTrainer):
    def __init__(self, sender: AbstractSender, trainer1: AbstractObjectDetectTrainer, trainer2: AbstractTrainer):
        super().__init__(sender)
        self.trainer1 = trainer1
        self.trainer2 = trainer2
        self.project_path1 = None
        self.project_path2 = None
        self.stage = None
        self.task_type = "Obj_Cls"

    def setupTrain(self, train_dir: str, is_resume: bool) -> Status:
        config_path = os.path.join(train_dir, "trainer.json")
        if not os.path.exists(config_path):
            return Status(success=False, error_str="no config")
        with open(config_path, 'r', encoding='utf-8') as f:
            train_config = json.load(f)
        if "export_dir" in train_config:
            self.export_dir = train_config["export_dir"]
            if not os.path.exists(self.export_dir):
                os.makedirs(self.export_dir)
        else:
            return Status(success=False, error_str="no export dir")

        self.project_path1 = os.path.join(train_dir, "stage1")
        self.project_path2 = os.path.join(train_dir, "stage2")

        trainer1_status = self.trainer1.setupTrain(self.project_path1, (is_resume and self.stage == "stage1"))
        trainer2_status = self.trainer2.setupTrain(self.project_path2, (is_resume and self.stage == "stage2"))

        if not trainer1_status.success:
            status = Status(success=False, error_str=trainer1_status.error_str)
            return status
        elif not trainer2_status.success:
            status = Status(success=False, error_str=trainer2_status.error_str)
            return status
        else:
            status = Status(success=True, error_str="")

        if not is_resume:
            self.start_epoch = 0
        else:
            if self.stage == "stage1":
                self.start_epoch = self.trainer1.start_epoch
            elif self.stage == "stage2":
                self.start_epoch = self.trainer1.end_epoch + self.trainer2.start_epoch
        self.end_epoch = self.trainer1.end_epoch + self.trainer2.end_epoch

        return status

    def trainEpoch(self, epoch: int) -> TrainStatus:
        if self.stop_signal:
            return
        train_status = TrainStatus()
        train_status.epoch = epoch
        self.epoch = epoch

        if epoch < self.trainer1.end_epoch:
            self.stage = "stage1"
            trainer1_status = self.trainer1.trainEpoch(epoch)
            train_status.train_loss = trainer1_status.train_loss
            train_status.val_accuracy = trainer1_status.val_accuracy
        else:
            self.stage = "stage2"
            trainer2_status = self.trainer2.trainEpoch(epoch - self.trainer1.end_epoch)
            train_status.train_loss = trainer2_status.train_loss
            train_status.val_accuracy = trainer2_status.val_accuracy

        return train_status

    def save_tmp_model(self):
        if self.stage == "stage1":
            self.trainer1.save_tmp_model()
        elif self.stage == "stage2":
            self.trainer2.save_tmp_model()

    # def load_tmp_model(self):
    #     raise NotImplemented

    def exportModel(self, export_path: str) -> bool:
        if self.stage == "stage1":
            export_path1 = os.path.join(export_path, "stage1")
            if not os.path.exists(export_path1):
                os.makedirs(export_path1)
            return self.trainer1.exportModel(export_path1)
        elif self.stage == "stage2":
            export_path1 = os.path.join(export_path, "stage1")
            export_path2 = os.path.join(export_path, "stage2")
            if not os.path.exists(export_path1):
                os.makedirs(export_path1)
            if not os.path.exists(export_path2):
                os.makedirs(export_path2)
            return self.trainer1.exportModel(export_path1) and self.trainer2.exportModel(export_path2)
        else:
            return False

