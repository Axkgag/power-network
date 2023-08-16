import os
import time
# from loguru import logger
import numpy as np
import cv2
import random
import warnings
import json

import sys
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
                    filemode='w',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from .lib.data.data_augment import ValTransform
from .lib.utils import postprocess, vis
from .lib.utils import(
    configure_nccl,
    fuse_model,
    get_local_rank,
    # get_model_info,
    setup_logger
)
from riemann_interface.defaults import AbstractObjectDetectPredictor
from typing import Dict, Any, List
from .lib.args import PredictArgs, EvalArgs, pretrain_model
from .lib.exp import Exp

def createInstance(config_path=None):
    predictor = YoloxPredictor()
    return predictor

class YoloxPredictor(AbstractObjectDetectPredictor):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    def setupPredictor(self, config: Dict[str, Any]):
        if config is None:
            return False
        self.config = config
        self.args = PredictArgs()
        self.exp = Exp()
        # self.device = self.args.device
        # self.fp16 = self.args.fp16
        self.preproc = ValTransform(legacy=self.args.legacy)

        keys = ["arch", "class_names", "input_res", "score_threshold", "iou_thr", "num_classes", "fp16", "gpus"]
        for key in keys:
            if key not in config:
                logger.error("{} is not in config".format(key))
                return False

        self.args.name = config["arch"]
        if self.args.name not in pretrain_model:
            logger.error("'{}' model does not exist".format(self.args.name))
            return False
        # self.args.patch_predict = config["patch_predict"]
        self.args.save_result = True

        # self.exp.data_dir = config["out_dir"]
        # self.exp.output_dir = config["out_dir"]
        self.exp.num_classes = config["num_classes"]
        self.exp.nmsthre = config["iou_thr"]
        self.exp.test_conf = config["score_threshold"]
        self.exp.test_size = config["input_res"]
        # self.exp.test_img = "test"
        self.exp.depth, self.exp.width = pretrain_model[self.args.name]

        if len(self.exp.test_conf) < self.exp.num_classes:
            num_to_fil = self.exp.num_classes - len(self.exp.test_conf)
            self.exp.test_conf.extend([0.5] * num_to_fil)
        elif len(self.exp.test_conf) > self.exp.num_classes:
            self.exp.test_conf = [0.5] * self.exp.num_classes

        self.cls_names = config["class_names"]
        self.num_classes = self.exp.num_classes
        self.confthre = self.exp.test_conf
        self.nmsthre = self.exp.nmsthre
        self.test_size = self.exp.test_size
        self.device = "cpu" if config["gpus"] == "-1" else "cuda:" + config["gpus"]
        self.fp16 = config["fp16"]
        # self.crop_stride = config["crop_stride"]
        # self.crop_size = config["crop_size"]
        self.use_patch = self.setCropConfig(config)

        # handler = logging.FileHandler(self.exp.output_dir + "/val_log.txt")
        # logger.addHandler(handler)
        return True

    def loadModel(self, model_dir: str) -> bool:
        predictor_config_path = os.path.join(model_dir, "predictor.json")
        ckpt = os.path.join(model_dir, "latest_ckpt.pth")
        if not os.path.exists(predictor_config_path):
            return False
        with open(predictor_config_path, 'r', encoding='utf-8') as pf:
            predictor_config = json.load(pf)
            pf.close()
        if not self.setupPredictor(predictor_config):
            return False
        # if self.args.save_result:
        #     self.vis_folder = os.path.join(self.exp.output_dir, "vis_res")
        #     os.makedirs(self.vis_folder, exist_ok=True)
        self.args.ckpt = ckpt

        self.model = self.exp.get_model()
        # logger.info("Model Summary: {}".format(get_model_info(self.model, self.test_size)))

        try:
            if self.device != "cpu":
                self.model.cuda()
                # self.model.to(self.device)
                # print(next(self.model.parrmeters()).device)
        except Exception as ex:
            print(ex)
            return False
        if self.fp16:
            self.model.half()
        self.model.eval()

        # if self.device == "gpu":
        #     self.model.cuda()
        #     if self.fp16:
        #         self.model.half()  # to FP16
        # self.model.eval()

        # if not self.args.trt:
        #     if self.args.ckpt is None:
        #         ckpt_file = os.path.join(model_dir, "latest_ckpt.pth")
        #     else:
        #         ckpt_file = self.args.ckpt
        #     logger.info("loading checkpoint")
        #     ckpt = torch.load(ckpt_file, map_location="cpu")
        #     # load the model state dict
        #     self.model.load_state_dict(ckpt["model"])
        #     logger.info("loaded checkpoint done.")

        if self.args.ckpt is None:
            ckpt_file = os.path.join(model_dir, "latest_ckpt.pth")
        else:
            ckpt_file = self.args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        self.model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

        # if self.args.fuse:
        #     logger.info("\tFusing model...")
        #     model = fuse_model(self.model)

        # if self.args.trt:
        #     assert not self.args.fuse, "TensorRT model is not support model fusing!"
        #     trt_file = os.path.join(model_dir, "model_trt.pth")
        #     assert os.path.exists(
        #         trt_file
        #     ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        #     model.head.decode_in_inference = False
        #     decoder = model.head.decode_outputs
        #     logger.info("Using TensorRT to inference")
        # else:
        #     trt_file = None
        #     decoder = None
        return True

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            # img = cv2.imread(img)
            img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        # if self.device == "gpu":
        #     img = img.cuda()
        #     if self.fp16:
        #         img = img.half()  # to FP16
        
        if self.device != "cpu":
            # img = img.to(self.device)
            img = img.cuda()
        if self.fp16:
            img = img.half()

        with torch.no_grad():
            # t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, 0.3,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, img_dir, save_dir, results_dict, current_time):
        results = results_dict["results"]
        # img = cv2.imread(img_dir)
        img = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), -1)
        result_image = vis(img, results, self.confthre, self.cls_names)

        if self.args.save_result:
            save_folder = os.path.join(
                save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(img_dir))
            # logger.info("Saving detection result in {}".format(save_file_name))
            # cv2.imwrite(save_file_name, result_image)
            cv2.imencode('.jpg', result_image)[1].tofile(save_file_name)

    def save_results(self, img_id, results_dict, outputs):
        results = results_dict["results"]

        for cls_name in results.keys():
            for bbox_score in results[cls_name]:
                bbox = [int(bbox_score[0]), int(bbox_score[1]),
                        int(bbox_score[2] - bbox_score[0]), int(bbox_score[3] - bbox_score[1])]
                score = bbox_score[4]
                cls_id = self.cls_names.index(cls_name) + 1
                outputs.append({
                    "bbox": bbox,
                    "score": score,
                    "image_id": img_id,
                    "category_id": cls_id
                })


    # def predict(self, img: Any, info: Dict[str, Any] = None) -> Dict[str, Any]:
    #     if self.args.patch_predict:
    #         return self.patchPredict(img, info)
    #     else:
    #         return self.normalPredict(img, info)

    def predict(self, img: Any, info: Dict[str, Any] = None) -> Dict[str, Any]:
        t0 = time.time()
        outputs, img_info = self.inference(img)
        ratio = img_info["ratio"]
        output = outputs[0]

        results = {}
        final_results = {"results": results}

        if output is None:
            return final_results
        output = output.cpu()

        for res in output:
            bbox = res[:4]
            bbox /= ratio
            score = res[4] * res[5]
            # cls = str(int(res[6]))
            cls = self.cls_names[int(res[6])]

            cls_id = int(res[6])
            if score < self.confthre[cls_id]:
                continue

            if cls not in results.keys():
                results[cls] = []
            results[cls].append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), float(score)])

        new_results = self.filter_by_iou(results)
        final_results["results"] = new_results

        # final_results["results"] = results

        # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return final_results

    # def patchPredict(self, img: Any, info: Dict[str, Any] = None) -> Dict[str, Any]:
    #     t0 = time.time()
    #
    #     crop_x, crop_y = self.crop_w, self.crop_h
    #     stride_x, stride_y = self.stride_x, self.stride_y
    #
    #     results = {}
    #     final_results = {"results": results}
    #
    #     # img_ori = cv2.imread(img)
    #     img_ori = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
    #     h, w = img_ori.shape[:2]
    #     w_r, h_r = int((w - crop_x) / stride_x + 1), int((h - crop_y) / stride_x + 1)
    #     H, W = crop_y + stride_y * (h_r - 1), crop_x + stride_x * (w_r - 1)
    #     img = cv2.resize(img_ori, dsize=(W, H))
    #     h_scale, w_scale = h / H, w / W
    #
    #     outputs = [None]
    #
    #     for i in range(h_r):
    #         for j in range(w_r):
    #             crop_h = i * stride_y
    #             crop_w = j * stride_x
    #             img_crop = img[crop_h: crop_h + crop_y, crop_w: crop_w + crop_y]
    #             crop_output, img_info = self.inference(img_crop)
    #
    #             if crop_output[0] is None:
    #                 continue
    #             else:
    #                 ret_crop = crop_output[0].cpu()
    #
    #             ratio = img_info["ratio"]
    #             ret_crop[:, :4] /= ratio
    #
    #             ret_crop[:, 0] += crop_w
    #             ret_crop[:, 1] += crop_h
    #             ret_crop[:, 2] += crop_w
    #             ret_crop[:, 3] += crop_h
    #
    #             if outputs[0] is None:
    #                 outputs[0] = ret_crop
    #             else:
    #                 outputs[0] = torch.cat((outputs[0], ret_crop), dim=0)
    #
    #     outputs = outputs[0].cpu()
    #     outputs[:, 0] *= w_scale
    #     outputs[:, 1] *= h_scale
    #     outputs[:, 2] *= w_scale
    #     outputs[:, 3] *= h_scale
    #
    #     conf_mask = (outputs[:, 4] * outputs[:, 5] >= self.confthre).squeeze()
    #     outputs = outputs[conf_mask]
    #     nms_index = self.nms_iou(outputs, self.nmsthre)
    #     outputs = outputs[nms_index]
    #
    #     for res in outputs:
    #         bbox = res[:4]
    #         score = res[4] * res[5]
    #         # cls = str(int(res[6]))
    #         cls = self.cls_names[int(res[6])]
    #         if cls not in results.keys():
    #             results[cls] = []
    #         results[cls].append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), float(score)])
    #     final_results["results"] = results
    #
    #     logger.info("Infer time: {:.4f}s".format(time.time() - t0))
    #     return final_results

    def get_image_list(self, path):
        IMAGE_EXT = [".jpg", ".JPG", ".jpeg", ".webp", ".bmp", ".png"]
        image_names = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in IMAGE_EXT:
                    image_names.append(apath)
        return image_names

    def filter_by_iou(self, results):
        def nms_iou(detections, iou_thr):
            x1, y1 = detections[:, 0], detections[:, 1]
            x2, y2 = detections[:, 2], detections[:, 3]
            areas = (y2 - y1 + 1) * (x2 - x1 + 1)
            nms_index = []
            index = areas.argsort()[::-1].copy()

            while index.size > 0:
                i = index[0]
                nms_index.append(i)
                x11 = np.maximum(x1[i], x1[index[1:]])
                y11 = np.maximum(y1[i], y1[index[1:]])
                x22 = np.minimum(x2[i], x2[index[1:]])
                y22 = np.minimum(y2[i], y2[index[1:]])

                w = np.maximum(0, x22 - x11 + 1)
                h = np.maximum(0, y22 - y11 + 1)
                overlaps = w * h

                ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)

                idx = np.where(ious <= iou_thr)[0]
                index = index[(idx + 1)]
            return nms_index

        new_results = {}
        bboxes = []
        cls_ids = {}
        anno = 0
        if len(results) == 0:
            return results
        for cls_id, bbox in results.items():
            bboxes.extend(list(bbox))
            new_results[cls_id] = []
            for _ in range(len(bbox)):
                cls_ids[anno] = cls_id
                anno += 1
        bboxes = np.stack(bboxes)
        keep_idx = nms_iou(bboxes, self.nmsthre)
        for idx in keep_idx:
            new_results[cls_ids[idx]].append(bboxes[idx])
        return new_results

    def filter_by_iou_plus(self, results):
        def nms_iou(detections, iou_thr):
            x1, y1 = detections[:, 0], detections[:, 1]
            x2, y2 = detections[:, 2], detections[:, 3]
            areas = (y2 - y1 + 1) * (x2 - x1 + 1)
            nms_index = []
            index = areas.argsort()[::-1].copy()

            while index.size > 0:
                i = index[0]
                nms_index.append(i)
                x11 = np.maximum(x1[i], x1[index[1:]])
                y11 = np.maximum(y1[i], y1[index[1:]])
                x22 = np.minimum(x2[i], x2[index[1:]])
                y22 = np.minimum(y2[i], y2[index[1:]])

                w = np.maximum(0, x22 - x11 + 1)
                h = np.maximum(0, y22 - y11 + 1)
                overlaps = w * h

                ious = overlaps / np.minimum(areas[i], areas[index[1:]])

                idx = np.where(ious <= iou_thr)[0]
                index = index[(idx + 1)]
            return nms_index

        new_results = {}
        bboxes = []
        cls_ids = {}
        anno = 0
        if len(results) == 0:
            return results
        for cls_id, bbox in results.items():
            bboxes.extend(list(bbox))
            new_results[cls_id] = []
            for _ in range(len(bbox)):
                cls_ids[anno] = cls_id
                anno += 1
        bboxes = np.stack(bboxes)
        keep_idx = nms_iou(bboxes, self.nmsthre)
        for idx in keep_idx:
            new_results[cls_ids[idx]].append(bboxes[idx])
        return new_results
