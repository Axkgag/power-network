import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import json
import numpy
import argparse

def get_annotations_and_images(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for child_of_root in root:
        if child_of_root.tag == 'Img_ID':
            img_id = child_of_root.text
        if child_of_root.tag == 'Img_FileFmt':
            img_fmt = child_of_root.text
        if child_of_root.tag == 'Img_SizeWidth':
            img_width = int(child_of_root.text)
        if child_of_root.tag == 'Img_SizeHeight':
            img_height = int(child_of_root.text)

        if child_of_root.tag == 'HRSC_Objects':
            annotations = []
            for child_item in child_of_root:
                if child_item.tag == 'HRSC_Object':
                    tmp_box = [0., 0., 0., 0., 0.]
                    for node in child_item:
                        if node.tag == 'Object_ID':
                            obj_id = int(node.text)
                        if node.tag == 'Class_ID':
                            cls_id = int(node.text)
                        if node.tag == 'mbox_cx':
                            tmp_box[0] = float(node.text)
                        if node.tag == 'mbox_cy':
                            tmp_box[1] = float(node.text)
                        if node.tag == 'mbox_w':
                            tmp_box[2] = float(node.text)
                        if node.tag == 'mbox_h':
                            tmp_box[3] = float(node.text)
                        if node.tag == 'mbox_ang':
                            ang = float(node.text)
                            ang = ang * 180 / numpy.pi
                            ang = ang if ang >= 0 else (ang + 180)
                            tmp_box[4] = int(ang)
                        area = tmp_box[2] * tmp_box[3]
                    tmp_box[0] = int(tmp_box[0] - tmp_box[2] / 2)
                    tmp_box[1] = int(tmp_box[1] - tmp_box[3] / 2)
                    tmp_box[2] = int(tmp_box[2])
                    tmp_box[3] = int(tmp_box[3])
                    annotations.append({
                        'area': area,
                        'bbox': tmp_box,
                        'category_id': cls_id,
                        'id': obj_id,
                        'image_id': img_id + "." + img_fmt,
                        'iscrowd': 0,
                        'segmentation': []
                    })
    img_info = {
        'file_name': img_id + "." + img_fmt,
        'width': img_width,
        'height': img_height
    }
    return annotations, img_info

def get_categories(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    categories = []
    categories_dict = {}
    for child_of_root in root:
        if child_of_root.tag == 'HRSC_Classes':
            for child_item in child_of_root:
                if child_item.tag == 'HRSC_Class':
                    for node in child_item:
                        if node.tag == 'Class_ID':
                            class_id = int(node.text)
                        if node.tag == 'Class_ShortName':
                            class_name = node.text
                    categories.append({
                        'id': class_id,
                        'name': class_name,
                        'supercategory': "none"
                    })
    for id, category in enumerate(categories):
        categories_dict[category["id"]] = id + 1
        category["id"] = id + 1
    return categories, categories_dict

def get_imges(img_path):
    img_list = os.listdir(img_path)
    images = []
    for id, name in enumerate(img_list):
        images.append({
            'data_captured': "",
            'file_name': name,
            'id': id,
            'width': 2592,
            'height': 1944,
            'license': 1
        })
    return images

def make_parser():
    parser = argparse.ArgumentParser("xml to coco")
    
    parser.add_argument("-d",
                        "--data_dir",
                        dest="data_dir",
                        default="./Train/",
                        type=str,
                        help="data dir")

    parser.add_argument("-i",
                        "--img_folder",
                        dest="img_folder",
                        default="AllImages",
                        type=str,
                        help="img folder")

    parser.add_argument("-x",
                        "--xml_folder",
                        dest="xml_folder",
                        default="Annotations",
                        type=str,
                        help="xml folder")

    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    img_path = os.path.join(args.data_dir, args.img_folder)
    xml_path = os.path.join(args.data_dir, args.xml_folder)
    file_name = os.path.join(args.data_dir, "annotations.json")
    images = get_imges(img_path)
    categories, categories_dict = get_categories(os.path.join(args.data_dir, "sysdata.xml"))
    annotations = []
    xml_list = os.listdir(xml_path)
    for img_id, xml_name in enumerate(xml_list):
        annotations_for_one_picture, img_info = get_annotations_and_images(join(xml_path, xml_name))
        for anno in annotations_for_one_picture:
            anno["image_id"] = img_id
            anno["category_id"] = categories_dict[anno["category_id"]]
        image = images[img_id]
        if image["file_name"] == img_info["file_name"]:
            image["width"] = img_info["width"]
            image["height"] = img_info["height"]
        annotations.extend(annotations_for_one_picture)
    for id, anno in enumerate(annotations):
        anno["id"] = id + 1
    info = {
        "contributor": "Riemann",
        "date_created": "2023-1-30",
        "description": "HRSC-2016",
        "url": "",
        "version": "v1.0",
        "year": "2023"
    }
    xml2coco = {
        "annotations": annotations,
        "categories": categories,
        "images": images,
        "info": info
    }
    with open(file_name, "w") as f:
        json.dump(xml2coco, f, indent=4)


