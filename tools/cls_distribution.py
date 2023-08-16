import argparse
import xml.etree.ElementTree as ET
import cv2
import os
from pycocotools.coco import COCO

def make_parse():
    parser = argparse.ArgumentParser("data distribution")

    parser.add_argument(
        "-d",
        "--data_dir",
        default="/data/gauss/lyh/datasets/power_networks/train/",
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
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    needs = ["010101021", "010401061"]

    coco = COCO(json_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    cat_dict = {}
    for cat_id in catIds:
        cat_info = coco.loadCats(cat_id)[0]
        cat_dict[cat_info["name"]] = []

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
            cat_dict[cat_name].append(img_name)
            if cat_name in needs:
                img = cv2.imread(os.path.join(img_path, img_name))
                cat_save_path = os.path.join(save_path, cat_name)
                if not os.path.exists(cat_save_path):
                    os.mkdir(cat_save_path)

                bbox = anno["bbox"]
                x0 = int(bbox[0])
                y0 = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                # crop_img = img[x0:(x0+w), y0:(y0+w), :].copy()
                crop_img = img[y0:(y0 + h), x0:(x0 + w), :].copy()
                # crop_img = img

                if crop_img.size == 0:
                    print(img_name, " :", [x0, y0, w, h])
                    continue
                crop_img_name = img_name.split(".")[0] + "_" + str(i) + ".jpg"
                cv2.imwrite(os.path.join(cat_save_path, crop_img_name), crop_img)
        if idx % 100 == 0:
            print("ok: ", idx)

    for cat_name, cat_dis in cat_dict.items():
        print(cat_name + ": " + str(len(cat_dis)))

