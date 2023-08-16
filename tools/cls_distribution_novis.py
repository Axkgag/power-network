import argparse
import xml.etree.ElementTree as ET
import cv2
import os
from pycocotools.coco import COCO
import statistics
import matplotlib.pyplot as plt

def make_parse():
    parser = argparse.ArgumentParser("data distribution")

    parser.add_argument(
        "-d",
        "--data_dir",
        default="/run/media/root/1000g/05-22/train/",
        type=str
    )

    parser.add_argument(
        "-i",
        "--img_folder",
        default="image",
        type=str
    )

    parser.add_argument(
        "-a",
        "--annotation",
        default="annotations.json",
        type=str
    )

    parser.add_argument(
        "-s",
        "--save_folder",
        default="data_distribution",
        type=str
    )

    return parser


if __name__ == "__main__":
    args = make_parse().parse_args()
    img_path = os.path.join(args.data_dir, args.img_folder)
    json_file = os.path.join(args.data_dir, args.annotation)
    save_path = os.path.join(args.data_dir, args.save_folder)
    save_results_dir = os.path.join(args.data_dir, "img_show")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_results_dir):
        os.mkdir(save_results_dir)

    coco = COCO(json_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    cat_dict = {}

    info_dict = {}

    for cat_id in catIds:
        cat_info = coco.loadCats(cat_id)[0]
        cat_dict[cat_info["name"]] = {
            "name": [],
            "short": [],
            "long": []
        }

    for idx in range(len(imgIds)):
        img_info = coco.loadImgs(imgIds[idx])[0]
        img_name = img_info["file_name"]
        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
        annos = coco.loadAnns(annIds)
        # img = cv2.imread(os.path.join(img_path, img_name))

        for i, anno in enumerate(annos):
            cls_id = anno["category_id"]
            cat_info = coco.loadCats(cls_id)[0]
            cat_name = cat_info["name"]
            cat_dict[cat_name]["name"].append(img_name)
            cat_save_path = os.path.join(save_path, cat_name)
            if not os.path.exists(cat_save_path):
                os.mkdir(cat_save_path)

            bbox = anno["bbox"]
            x0 = int(bbox[0])
            y0 = int(bbox[1])
            w = int(bbox[2])
            h = int(bbox[3])
            cat_dict[cat_name]["short"].append(min(w, h))
            cat_dict[cat_name]["long"].append(max(w, h))
            # crop_img = img[y0:(y0 + h), x0:(x0 + w), :].copy()
            # # crop_img = img
            # if crop_img.size == 0:
            #     print(img_name, " :", [x0, y0, w, h])
            #     continue
            # crop_img_name = img_name.split(".")[0] + "_" + str(i) + ".jpg"  # TODO
            # cv2.imwrite(os.path.join(cat_save_path, crop_img_name), crop_img)
        if idx % 100 == 0:
            print("ok: ", idx)

    short_list = []
    long_list = []

    for cat_name in cat_dict.keys():
        name = cat_dict[cat_name]['name']
        short = cat_dict[cat_name]['short']
        long = cat_dict[cat_name]['long']

        short_list += short
        long_list += long

        print(cat_name + ": " + str(len(name)))

        if len(short) > 0 and len(long) > 0:
            print("short: max: {}, min: {}, mean: {}, mid: {}".format(max(short), min(short), int(statistics.mean(short)), statistics.median(short)))
            print("long: max: {}, min: {}, mean: {}, mid: {} ".format(max(long), min(long), int(statistics.mean(long)), statistics.median(long)))

    #     with open(os.path.join(save_results_dir, "{}.txt".format(cat_name)), 'w') as f:
    #         f.write(" ".join(map(str, short)) + "\n")
    #         f.write(" ".join(map(str, long)) + "\n")
    #
    # with open(os.path.join(save_results_dir, "all.txt"), 'w') as f:
    #     f.write(" ".join(map(str, short_list)) + "\n")
    #     f.write(" ".join(map(str, long_list)) + "\n")