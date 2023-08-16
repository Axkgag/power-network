import xml.etree.ElementTree as ET
import xml.dom.minidom as xmldom
import pickle
import os
from os import listdir, getcwd
from os.path import join
import json
import numpy
import argparse


def get_catlist(txt_path):
    with open(txt_path) as lf:
        cat_lines = lf.readlines()
        lf.close()
    # cat_dict1 = {}
    cat_dict = {}
    for id, cat_line in enumerate(cat_lines):
        cat_name = cat_line.split(' ')[0]
        # cat_dict1[id] = cat_line[:-1]
        # cat_dict[cat_line[:-1]] = id + 1
        cat_dict[cat_name] = id + 1
    return cat_dict

def get_images(xml_path):
    xml_list = os.listdir(xml_path)
    xml_list.sort()
    i = 1
    images = []
    annotations = []
    cat_list =[]
    for img_id, xml_name in enumerate(xml_list):
        tree = ET.parse(os.path.join(xml_path, xml_name))
        img_name = tree.find("filename").text
        img_w = int(tree.find("size").find("width").text)
        img_h = int(tree.find("size").find("height").text)
        images.append({
            "id": img_id,
            "file_name": img_name,
            "width": img_w,
            "height": img_h,
            "date_captured": ""
        })
        objs = tree.findall("object")
        for obj in objs:
            component = obj.find("Component").text
            name = obj.find("name").text
            part = obj.find("Part").text
            dedescription = obj.find("DeDescription").text
            cat_name = component + '-' + name + '-' + part + '-' + dedescription
            if cat_name not in cat_list:
                cat_list.append(cat_name)
                cat_id = len(cat_list) - 1
            else:
                cat_id = cat_list.index(cat_name)
            xmin = float(obj.find("bndbox").find("xmin").text)
            ymin = float(obj.find("bndbox").find("ymin").text)
            xmax = float(obj.find("bndbox").find("xmax").text)
            ymax = float(obj.find("bndbox").find("ymax").text)
            w = int(abs(xmax - xmin))
            h = int(abs(ymax - ymin))
            x = min(xmax, xmin)
            y = min(ymax, ymin)
            if w == 0 or h == 0:
                print(img_name, ":", x, y, w, h)
                continue
            bbox = [int(x), int(y), w, h]
            annotations.append({
                "id": i,
                "area": w * h,
                "bbox": bbox,
                "category_id": cat_id,
                "image_id": img_id,
                "iscrowd": 0,
                "segmentation": []
            })
            i += 1
    return images, annotations, cat_list


if __name__ == "__main__":
    txt_path = "/run/media/root/1000g/05-22/train/class_name.txt"
    xml_path = "/run/media/root/1000g/05-22/train/xml"
    output_dir = "/run/media/root/1000g/05-22/train/annotations.json"
    images = []
    categories = []
    annotations = []

    dict = get_catlist(txt_path)
    for key, value in dict.items():
        categories.append({
            "id": value,
            "name": key,
            "supercategory": "none"
        })

    images, annotations, cat_list = get_images(xml_path)
    # unicode_txt = "\u67b6\u7a7a\u7ebf\u8def-\u5bfc\u7ebf-\u5bfc\u7ebf-\u7edd\u7f18\u62a4\u5957\u635f\u574f"
    # utf_txt = unicode_txt.encode("unicode_escape").decode("utf-8")
    # cat_list.append(unicode_txt)
    for cat_id, cat_name in enumerate(cat_list):
        categories.append({
            "id": cat_id,
            "name": cat_name,
            "supercategory": "none"
        })
    info = {
        "contributor": "Riemann",
        "date_created": "2022-10-09_16:54:08",
        "description": "Exported from RIVIP-Learn",
        "url": "",
        "version": "1",
        "year": "2022"
    }
    json_file = {
        "categories": categories,
        "images": images,
        "annotations": annotations,
        "info": info
    }

    with open(txt_path, 'w', encoding='utf-8') as tf:
        for cat_name in cat_list:
            tf.write(cat_name + '\n')

    with open(output_dir, 'w', encoding='utf-8') as jf:
        json.dump(json_file, jf, indent=4)
