import numpy as np
import math
import pycocotools.coco as coco
import os
import cv2
import torch
import json
import copy

from ..utils.boxes import postprocess, convert_box

class REvaluator:
    def __init__(
            self,
            # dataloader,
            img_size,
            data_dir,
            val_path,
            anno_name,
            class_names,
            confthre,
            nmsthre,
            preproc,
    ):
        # self.dataloader = dataloader
        self.data_dir = data_dir
        self.val_path = os.path.join(self.data_dir, val_path)
        self.coco_anno = coco.COCO(os.path.join(self.data_dir, anno_name))
        self.file_names = [item["file_name"] for item in self.coco_anno.loadImgs(ids=self.coco_anno.getImgIds())]
        self.class_names = class_names
        self.res_save_dir = os.path.join(self.data_dir, "cache/txt_predict")
        self.gt_save_dir = os.path.join(self.data_dir, "cache/txt_gt")
        self.num_classes = len(self.class_names)
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.img_size = img_size  # test_size
        self.preproc = preproc
        cache_dir = os.path.join(self.data_dir, "cache")
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

    def dump_gt(self):
        if not os.path.exists(self.gt_save_dir):
            os.mkdir(self.gt_save_dir)

        converted_gt = {}
        for file_name in self.file_names:
            file_name = file_name.split(".")[0]
            converted_gt[file_name] = []
        for i, img_id in enumerate(self.coco_anno.getImgIds()):
            annIds = self.coco_anno.getAnnIds(imgIds=[img_id])
            anns = self.coco_anno.loadAnns(annIds)
            file_name = self.file_names[i].split(".")[0]
            for ann in anns:
                cat_id = self.class_names[ann["category_id"] - 1]
                x, y, w, h, ang = ann["bbox"]  # (x, y)检测框左上角坐标
                bbox_score = [x, y, x + w, y + h, ang] + [1, 1]
                score, x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = convert_box(bbox_score)
                converted_gt[file_name].append(str(x0n) + " " + str(y0n) + " " +
                                               str(x1n) + " " + str(y1n) + " " +
                                               str(x2n) + " " + str(y2n) + " " +
                                               str(x3n) + " " + str(y3n) + " " +
                                               cat_id + " " + "0" + "\n")
        for file_name in self.file_names:
            file_name = file_name.split(".")[0]
            with open(os.path.join(self.gt_save_dir, file_name+".txt"), 'w') as f:
                f.writelines(converted_gt[file_name])

        with open(os.path.join(self.data_dir, "cache/valset.txt"), "w") as ff:
            for name in self.file_names:
                ff.write(name.split(".")[0] + " " + "\n")

    def get_image_list(self, path):
        IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
        image_names = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in IMAGE_EXT:
                    image_names.append(apath)
        return image_names

    def Predict(self, model, img, half):
        outputs, img_info = self.inference(model, img, half)
        ratio = img_info["ratio"]
        output = outputs[0]

        results = {}

        if output is None:
            return results
        output = output.cpu()

        for res in output:
            res[:4] /= ratio
            bbox = res[:5]
            score = res[5] * res[6]
            cls = int(res[7])
            if cls not in results.keys():
                results[cls] = []
            results[cls].append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4]), float(score)])
        return results

    def inference(self, model, img, half=False):
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

        ratio = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.img_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if half:
            img = img.half()
        img = img.cuda()

        with torch.no_grad():
            outputs = model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

    def get_results(self, model, half=False, save_dir=None):
        if not os.path.exists(self.res_save_dir):
            os.mkdir(self.res_save_dir)
        # if save_dir:
            # save_dir = os.path.join(save_dir, "train_results")
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)
        train_results = []
        imgs_list = self.coco_anno.loadImgs(self.coco_anno.getImgIds())
        imgs_dict = {}
        for img_info in imgs_list:
            imgs_dict[img_info["file_name"]] = img_info["id"]


        converted_results = {}
        for class_name in self.class_names:
            converted_results[class_name] = []

        # img_path = os.path.join(self.data_dir, "val")
        if os.path.isdir(self.val_path):
            files = self.get_image_list(self.val_path)
            files.sort()
            for img_name in files:
                results = self.Predict(model, img_name, half)
                img_id = imgs_dict[os.path.basename(img_name)]
                for cls_id in results.keys():
                    for bbox_score in results[cls_id]:
                        if save_dir:
                            res = copy.deepcopy(bbox_score)
                            res[2] -= res[0]
                            res[3] -= res[1]
                            train_res = {
                                "image_id": img_id,
                                "category_id": cls_id,
                                "bbox": res[:5],
                                "score": res[5]
                            }
                            train_results.append(train_res)

                        score, x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = convert_box(bbox_score)
                        if score < self.confthre:
                            continue
                        converted_results[self.class_names[cls_id]].append(
                            os.path.basename(img_name).split(".")[0] + " " +
                            str(score) + " " + str(x0n) + " " + str(y0n) + " " +
                            str(x1n) + " " + str(y1n) + " " +
                            str(x2n) + " " + str(y2n) + " " +
                            str(x3n) + " " + str(y3n) + "\n")
            for class_name in self.class_names:
                with open(os.path.join(self.res_save_dir, class_name + ".txt"), "w") as f:
                    f.writelines(converted_results[class_name])
            if save_dir:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                with open(os.path.join(save_dir, "train_results.json"), 'w') as jf:
                    json.dump(train_results, jf)
                # if save_dir:
                #     with open(os.path.join(save_dir, class_name + ".txt"), "w") as sf:
                #         sf.writelines(converted_results[class_name])

    # def dump_results(self, model, half=False):
    #     if not os.path.exists(self.res_save_dir):
    #         os.mkdir(self.res_save_dir)
    #
    #     converted_results = {}
    #     for class_name in self.class_names:
    #         converted_results[class_name] = []
    #
    #     tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
    #     model = model.eval()
    #     if half:
    #         model = model.half()
    #
    #     for imgs, _, info_imgs, _, img_names in self.dataloader:
    #         r = info_imgs[2][0].item()
    #         with torch.no_grad():
    #             for img, _, img_name in zip(imgs, info_imgs, img_names):
    #                 img = img.type(tensor_type).unsqueeze(0)
    #                 outputs = model(img)
    #
    #                 outputs = postprocess(
    #                     outputs, self.num_classes, self.confthre, self.nmsthre
    #                 )
    #                 # outputs = (x1, y1, x2, y2, ang, obj_conf, class_conf, class_pred)
    #                 outputs = outputs[0]
    #                 if outputs is None:
    #                     continue
    #                 outputs = outputs.cpu()
    #
    #                 # conf_mask = (outputs[:, 5] * outputs[:, 6] >= self.confthre).squeeze()
    #                 # outputs = outputs[conf_mask]
    #                 # nms_index = self.nms_iou(outputs, self.nmsthre)
    #                 # outputs = outputs[nms_index]
    #
    #                 for res in outputs:
    #                     res = res.numpy()
    #                     bbox_score = res[0:7]
    #                     # bbox_score[:4] /= scale
    #                     bbox_score[:4] = bbox_score[:4] / r
    #                     cat_id = int(res[7])
    #                     score, x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n = convert_box(bbox_score)
    #                     if score < self.confthre:
    #                         continue
    #                     converted_results[self.class_names[cat_id]].append(
    #                         img_name.split(".")[0] + " " +
    #                         str(score) + " " + str(x0n) + " " + str(y0n) + " " +
    #                         str(x1n) + " " + str(y1n) + " " +
    #                         str(x2n) + " " + str(y2n) + " " +
    #                         str(x3n) + " " + str(y3n) + "\n")
    #     for class_name in self.class_names:
    #         with open(os.path.join(self.res_save_dir, class_name+".txt"), "w") as f:
    #             f.writelines(converted_results[class_name])

    def nms_iou(self, detections, iou_thr):
        x1, y1 = detections[:, 0], detections[:, 1]
        x2, y2 = detections[:, 2], detections[:, 3]
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        nms_index = []
        index = areas.numpy().argsort()[::-1].copy()

        while index.size > 0:
            i = index[0]
            nms_index.append(i)
            x11 = torch.max(x1[i], x1[index[1:]])
            y11 = torch.max(y1[i], y1[index[1:]])
            x22 = torch.min(x2[i], x2[index[1:]])
            y22 = torch.min(y2[i], y2[index[1:]])

            w = torch.max(torch.tensor(0), x22 - x11 + 1)
            h = torch.max(torch.tensor(0), y22 - y11 + 1)
            overlaps = w * h

            ious = overlaps / torch.min(areas[i], areas[index[1:]])

            idx = np.where(ious <= iou_thr)[0]
            index = index[(idx + 1)]
        return nms_index

    def evaluate(self, model, half=False, save_dir=None):
        from .voc_eval import voc_eval

        # self.dump_results(model, half=half)
        self.get_results(model, half=half, save_dir=save_dir)

        classAPs = {}
        mAP = 0
        for class_name in self.class_names:
            rec, prec, ap = voc_eval(
                detpath=self.res_save_dir + r'/{:s}.txt',  # predict_results save dir
                annopath=self.gt_save_dir + r'/{:s}.txt',  # gt_bbox save dir
                imagesetfile=self.data_dir + r'/cache/valset.txt',  # img_info save dir
                classname=class_name,
                ovthresh=0.5,
                use_07_metric=True
            )
            mAP += ap
            classAPs[class_name] = ap
        mAP /= self.num_classes

        return mAP, classAPs
