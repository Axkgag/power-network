import datetime
import os
import json
import time
# from loguru import logger

import sys
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
                    filemode='w',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


from torch import nn
from .lib.models.network_blocks import SiLU
from .lib.utils import replace_module

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.tensorboard import SummaryWriter

from .lib.args import TrainArgs, pretrain_model
from .lib.data import DataPrefetcher
from .lib.exp import Exp
from .lib.utils import (
    MeterBuffer,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)

from riemann_interface.defaults import Status, TrainStatus, AbstractSender, AbstractObjectDetectTrainer

class YoloxTrainer(AbstractObjectDetectTrainer):
    def __init__(self, sender: AbstractSender, config_name: str):
        super().__init__(sender)
        self.config_name = config_name 

    def setupInfo(self, train_dir: str) -> bool:
        self.exp = Exp()

        trainer_config_path = os.path.join(train_dir, self.config_name)
        with open(trainer_config_path, 'r', encoding='utf-8') as f:
            train_config = json.load(f)

            keys = ["arch", "batch_size", "class_names", "input_res", "gpus", "num_classes", "num_epochs",
                    "num_workers", "score_threshold", "iou_thr", "val_intervals", "patch_train", "export_dir",
                    "crop_w", "crop_h", "stride_x", "stride_y"]
            for key in keys:
                if key not in train_config:
                    logger.error("{} is not in config".format(key))
                    return False

            self.config = train_config
            logger.info(self.config)
            train_anno_path = os.path.join(train_dir, "annotations", self.config["train_anno"])
            train_img_path = os.path.join(train_dir, self.config["train_img"])
            val_anno_path = os.path.join(train_dir, "annotations", self.config["val_anno"])
            val_img_path = os.path.join(train_dir, self.config["val_img"])

            self.exp.data_dir = train_dir
            self.arch = self.config["arch"]
            if self.arch not in pretrain_model:
                logger.error("'{}' model does not exist".format(self.arch))
                return False
            self.exp.depth, self.exp.width = pretrain_model[self.arch]
            self.ckpt = "./weights/{}.pth".format(self.config["ckpt_model"])
            self.batch_size = self.config["batch_size"]
            self.devices = int(self.config["gpus"])
            self.patch_train = self.config["patch_train"]

            self.fp16 = False
            self.cache = False
            self.occupy = False
            self.dynamic = False
            self.decode_in_inference = False
            self.no_onnxsim = False

            self.exp.nmsthre = self.config["iou_thr"]
            self.exp.num_classes = self.config["num_classes"]
            self.exp.input_size = tuple(self.config["input_res"])
            # self.exp.test_conf = self.config["score_threshold"]
            self.exp.max_epoch = self.config["num_epochs"]
            self.exp.data_num_workers = self.config["num_workers"]
            self.exp.eval_interval = self.config["val_intervals"]
            self.export_dir = self.config["export_dir"]
            self.exp.train_img = self.config["train_img"]
            self.exp.train_ann = self.config["train_anno"]
            self.exp.val_img = self.config["val_img"]
            self.exp.val_ann = self.config["val_anno"]

            if len(self.config["score_threshold"]) < self.exp.num_classes:
                num_to_fil = self.exp.num_classes - len(self.config["score_threshold"])
                self.config["score_threshold"].extend([0.5] * num_to_fil)
            elif len(self.config["score_threshold"]) > self.exp.num_classes:
                self.config["score_threshold"] = [0.5] * self.exp.num_classes


            if self.config["patch_train"]:
                self.crop_w = self.config["crop_w"]
                self.crop_h = self.config["crop_h"]
                self.stride_x = self.config["stride_x"]
                self.stride_y = self.config["stride_y"]

                crop_train_img_path = os.path.join(train_dir, "crop_" + self.config["train_img"])
                crop_train_anno_path = os.path.join(train_dir, "annotations/crop_" + self.config["train_anno"])
                if not os.path.exists(crop_train_img_path) or not os.path.exists(crop_train_anno_path):
                    self.setupDataset(train_anno_path, train_img_path, crop_train_img_path, crop_train_anno_path)
                self.exp.train_ann = "crop_" + self.config["train_anno"]
                self.exp.train_img = "crop_" + self.config["train_img"]

                if self.exp.eval_interval > 0:
                    crop_val_img_path = os.path.join(train_dir, "crop_" + self.config["val_img"])
                    crop_val_anno_path = os.path.join(train_dir, "annotations/crop_" + self.config["val_anno"])
                    if not os.path.exists(crop_val_img_path) or not os.path.exists(crop_val_anno_path):
                        self.setupDataset(val_anno_path, val_img_path, crop_val_img_path, crop_val_anno_path)
                    self.exp.val_ann = "crop_" + self.config["val_anno"]
                    self.exp.val_img = "crop_" + self.config["val_img"]

        self.epoch = 0
        self.end_epoch = self.exp.max_epoch
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cpu" if self.devices == -1 else "cuda:{}".format(self.devices)
        self.use_model_ema = self.exp.ema

        self.data_type = torch.float16 if self.fp16 else torch.float32
        self.input_size = self.exp.input_size
        self.best_ap = 0

        self.meter = MeterBuffer(window_size=self.exp.print_interval)

        cache_dir = os.path.join(self.export_dir, "cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

        self.task_type = "object_detection"

        handler = logging.FileHandler(self.cache_dir + "/train_log.txt")
        logger.addHandler(handler)

        logging.info("train dir: {}".format(self.exp.data_dir))
        logging.info("train images folder: {}".format(self.exp.train_img))
        logging.info("train annotations name: {}".format(self.exp.train_ann))
        logging.info("val images folder: {}".format(self.exp.val_img))
        logging.info("val annotations name: {}".format(self.exp.val_ann))

        return True

    def setupTrain(self, train_dir: str, is_resume: bool = False) -> Status:
        if not os.path.exists(train_dir):
            logging.error("train_dir is not exist")
        logging.info("train_dir: {}".format(train_dir))
        if not self.setupInfo(train_dir):
            status = Status(success=False, error_str="setupInfo fail")
            return status
        logging.info("Creating model...")

        self.is_resume = is_resume

        # logger.info("args: {}".format(self.args))
        # logger.info("exp value:\n{}".format(self.exp))
        if self.devices != -1:
            torch.cuda.set_device(self.devices)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        self.optimizer = self.exp.get_optimizer(self.batch_size)

        model = self.resume_train(model)

        self.no_aug = self.start_epoch >= self.exp.max_epoch - self.exp.no_aug_epochs
        # self.no_aug = True
        
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.cache
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.batch_size, self.max_iter
        )
        if self.occupy:
            occupy_mem(5)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.exp.ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        if self.exp.eval_interval > 0:
            self.evaluator = self.exp.get_evaluator(
                batch_size=self.batch_size, is_distributed=self.is_distributed
            )

        logger.info("Training start...")
        # logger.info("\n{}".format(model))

        status = Status(success=True, error_str="")
        # print("finish setup train start at", self.start_epoch)
        return status

    def trainEpoch(self, epoch: int) -> TrainStatus:
        if self.stop_signal:
            return
        train_status = TrainStatus()
        train_status.epoch = epoch
        self.epoch = epoch
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        # if epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
        if epoch + 1 == self.exp.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            # logger.info("--->No mosaic aug now!")
            # self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            if self.exp.eval_interval > 0:
                self.exp.eval_interval = 1
            # if not self.no_aug:
            #     self.save_ckpt(ckpt_name="last_mosaic_epoch")

        # 2.train_in_iter
        for self.iter in range(self.max_iter):
            # 2.1 before_iter
            # 2.2 train_one_iter
            iter_start_time = time.time()

            inps, targets = self.prefetcher.next()
            inps = inps.to(self.data_type)
            targets = targets.to(self.data_type)
            targets.requires_grad = False
            inps, targets = self.exp.preprocess(inps, targets, self.input_size)
            data_end_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                outputs = self.model(inps, targets)

            loss = outputs["total_loss"]
            train_status.train_loss = loss.item()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.use_model_ema:
                self.ema_model.update(self.model)

            lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            iter_end_time = time.time()
            self.meter.update(
                iter_time=iter_end_time - iter_start_time,
                data_time=data_end_time - iter_start_time,
                lr=lr,
                **outputs,
            )

            # 2.3 after_iter
            if (self.iter + 1) % self.exp.print_interval == 0:
                # TODO check ETA logic
                left_iters = self.max_iter * self.exp.max_epoch - (self.progress_in_iter + 1)
                eta_seconds = self.meter["iter_time"].global_avg * left_iters
                eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

                progress_str = "epoch: {}/{}, iter: {}/{}".format(
                    self.epoch + 1, self.exp.max_epoch, self.iter + 1, self.max_iter
                )
                loss_meter = self.meter.get_filtered_meter("loss")
                loss_str = ", ".join(
                    ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
                )

                time_meter = self.meter.get_filtered_meter("time")
                time_str = ", ".join(
                    ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
                )

                logger.info(
                    "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                        progress_str,
                        gpu_mem_usage(),
                        time_str,
                        loss_str,
                        self.meter["lr"].latest,
                    )
                    + (", size: {:d}, {}".format(self.input_size[0], eta_str))
                )

                self.meter.clear_meters()

            if (self.progress_in_iter + 1) % 10 == 0:
                self.input_size = self.exp.random_resize(
                    self.train_loader, self.epoch, self.rank, self.is_distributed
                )

        # 3. after_epoch
        self.save_ckpt(ckpt_name="last_epoch")
        if self.exp.eval_interval > 0 and (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            ap50_95 = self.evaluate_and_save_model()
            train_status.val_accuracy = ap50_95

        return train_status

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.is_resume:
            model = self.load_tmp_model(model)
        else:
            if self.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)

        if self.rank == 0:
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.exp.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

        return ap50_95

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None, export_dir=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model

            if export_dir is None:
                export_dir = self.cache_dir

            logger.info("Save {} to {}".format(ckpt_name + "_ckpt.pth", export_dir))

            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
                "curr_ap": ap,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                export_dir,
                ckpt_name,
            )


    def save_results(self, predictions, save_dir):
        results = []
        for img_id in predictions.keys():
            img_results = predictions[img_id]
            num_bboxes = len(img_results['bboxes'])
            for i in range(num_bboxes):
                bbox = img_results["bboxes"][i]
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                results.append({
                    "image_id": img_id,
                    "category_id": img_results["categories"][i],
                    "bbox": bbox,
                    "score": img_results["scores"][i]
                })
        if len(results) == 0:
            results.append({
                "image_id": 0,
                "category_id": 0,
                "bbox": [0, 0, 0, 0],
                "score": 0.0
            })
        if not os.path.exists(os.path.join(save_dir, "report")):
            os.makedirs(os.path.join(save_dir, "report"))
        with open(os.path.join(save_dir, "report","train_results.json"), 'w') as f:
            json.dump(results, f)

    def testTrainDataset(self, save_dir, train_mode):
        from .lib.data import COCODataset, ValTransform
        from .lib.evaluators import COCOEvaluator

        if train_mode == "train":
            data_dir = self.exp.data_dir
            json_file = self.exp.train_ann
            name = self.exp.train_img
        elif train_mode == "val":
            data_dir = self.val_dir
            json_file = "annotations.json"
            name = "image"
        elif train_mode == "test":
            data_dir = self.test_dir
            json_file = "annotations.json"
            name = "image"

        testTrain_datasets = COCODataset(
            data_dir=data_dir,
            json_file=json_file,
            name=name,
            img_size=self.exp.test_size,
            preproc=ValTransform(legacy=False),
        )
        testTrain_loader = torch.utils.data.DataLoader(
            testTrain_datasets,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        train_evaluator = COCOEvaluator(
            dataloader=testTrain_loader,
            img_size=self.exp.test_size,
            confthre=self.exp.test_conf,
            nmsthre=self.exp.nmsthre,
            num_classes=self.exp.num_classes,
            testdev=False
        )
        testmodel = self.exp.get_model()
        torch.cuda.set_device(self.devices)
        testmodel.to(self.device)
        testmodel.eval()
        ckpt = torch.load(os.path.join(save_dir, "latest_ckpt.pth"), map_location=self.device)
        testmodel.load_state_dict(ckpt["model"])
        stat = {}

        with adjust_status(testmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                testmodel, train_evaluator, self.is_distributed, return_outputs=True
            )
        logger.info("\n" + summary)
        self.save_results(predictions, save_dir)

        # import pycocotools.coco as coco
        # from utils.riemann_coco_eval import actualGetAP
        # coco_anno = coco.COCO(os.path.join(self.exp.data_dir, self.exp.train_ann))
        # coco_dets = coco_anno.loadRes(os.path.join(save_dir, "report", "train_results.json"))
        # stat = actualGetAP(gt_anno=coco_anno, dets_anno=coco_dets, iou_type="bbox")

        # stat["ap50_90"] = ap50_95
        # stat["ap50"] = ap50
        return stat

    def exportModel(self, export_path: str) -> bool:
        self.save_ckpt(ckpt_name="latest", export_dir=export_path)

        with open(os.path.join(export_path, "predictor.json"), "w", encoding='utf-8') as pf:
            pred_json = {
                "arch": self.config["arch"],
                "class_names": self.config["class_names"],
                "input_res": self.config["input_res"],
                "score_threshold": self.config["score_threshold"],
                "iou_thr": self.config["iou_thr"],
                "num_classes": self.config["num_classes"],
                "fp16": False,
                "gpus": str(self.config["gpus"]),
            }
            if self.config["patch_train"]:
                pred_json["crop_w"] = self.config["crop_w"]
                pred_json["crop_h"] = self.config["crop_h"]
                pred_json["stride_x"] = self.config["stride_x"]
                pred_json["stride_y"] = self.config["stride_y"]
            json.dump(pred_json, pf, indent=4, ensure_ascii=False)

        # status = []
        # datasets = []
        # train_status = self.testTrainDataset(export_path, "train")
        # status.append(train_status)
        # datasets.append(1)
        # if self.val_dir != "":
        #     val_status = self.testTrainDataset(export_path, "val")
        #     status.append(val_status)
        #     datasets.append(2)
        # if self.test_dir != "":
        #     test_status = self.testTrainDataset(export_path, "test")
        #     status.append(test_status)
        #     datasets.append(3)
        # results = {"stats": status, "datasets": datasets, "task_type": "ObjectDetection"}
        # self.sender.send(method="train_result", data=results)

        return True

    def exportOnnx(self, model_dir: str, export_path: str):
        model = self.exp.get_model()
        ckpt_file = os.path.join(model_dir, "latest_ckpt.pth")
        if not os.path.exists(ckpt_file):
            logger.error("ckpt is not exist !")
            return

        ckpt = torch.load(ckpt_file, map_location="cpu")

        model.eval()
        if "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt)
        model = replace_module(model, nn.SiLU, SiLU)
        model.head.decode_in_inference = self.decode_in_inference

        logger.info("loading checkpoint done.")
        dummy_input = torch.randn(1, 3, self.exp.test_size[0], self.exp.test_size[1]).cuda()

        torch.onnx._export(
            model,
            dummy_input,
            os.path.join(export_path, "model.onnx"),
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch'},
                'output': {0: 'batch'},
            } if self.dynamic else None,
            opset_version=12,
        )
        logger.info("generated onnx model named {}".format("model.onnx"))

        if not self.no_onnxsim:
            import onnx
            from onnxsim import simplify
            input_shapes = {"images": list(dummy_input.shape)} if self.dynamic else None
            onnx_model = onnx.load(os.path.join(export_path, "model.onnx"))
            model_simp, check = simplify(onnx_model,
                                         dynamic_input_shape=self.dynamic,
                                         input_shapes=input_shapes)
            onnx.save(model_simp, os.path.join(export_path, "model.onnx"))
            logger.info("generated simplified onnx model named {}".format("model.onnx"))


    def save_tmp_model(self):
        self.save_ckpt(ckpt_name="stop")
        logger.info("have saved model: stop_ckpt.pth")

    def load_tmp_model(self, model):
        ckpt_file = os.path.join(self.cache_dir, "last_epoch_ckpt.pth")
        if not os.path.exists(ckpt_file):
            ckpt_file = self.ckpt
        ckpt = torch.load(ckpt_file, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_ap = ckpt.pop("best_ap", 0)
        self.start_epoch = ckpt["start_epoch"]
        logger.info(
            "loaded checkpoint '{}' (epoch {}) ".format(
                self.is_resume, self.start_epoch
            )
        )
        return model

    def verify_onnx_model(self, model_path, onnx_model_path):
        import onnxruntime as rt
        import numpy as np

        sess = rt.InferenceSession(os.path.join(onnx_model_path, "model.onnx"))
        model = self.exp.get_model()
        model.eval()
        model.to(self.device)
        ckpt = torch.load(os.path.join(model_path, "latest_ckpt.pth"), map_location=self.device)["model"]
        model = load_ckpt(model, ckpt)
        model = replace_module(model, nn.SiLU, SiLU)
        model.head.decode_in_inference = self.decode_in_inference

        diff = 0.0
        for i in range(10):
            x = torch.rand((1, 3, self.exp.test_size[0], self.exp.test_size[1]))
            model_output = model(x.cuda())

            onnx_input = {sess.get_inputs()[0].name: x.numpy()}
            onnx_output = sess.run(None, onnx_input)

            model_output = model_output.detach().cpu().numpy()

            diff += np.mean(np.abs(model_output - onnx_output[0]))
        print("error: ", diff / 10)
