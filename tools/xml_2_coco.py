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
        # cat_code = cat_line.split(' ')[1]
        # # cat_dict1[id] = cat_line[:-1]
        # cat_dict[cat_code[:-1]] = id + 1

        cat_name = cat_line.split(' ')[0]
        cat_code = cat_line.split(' ')[1]
        cat_id = cat_line.split(' ')[2][:-1]
        cat_dict[cat_name] = [cat_code, cat_id]
    return cat_dict

def get_images(xml_path, cat_dict):
    xml_list = os.listdir(xml_path)
    xml_list.sort()
    i = 1
    images = []
    annotations = []
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
            cat_code = obj.find("code").text
            if cat_name not in cat_dict:
                # cat_id = 6
                continue
            else:
                cat_id = cat_dict[cat_name][1]
                # cat_id = 1
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
                "category_id": int(cat_id),
                "image_id": img_id,
                "iscrowd": 0,
                "segmentation": []
            })
            i += 1
    return images, annotations


if __name__ == "__main__":
    txt_path = "/data/gauss/lyh/datasets/power_networks/test/class_name_yolox.txt"
    xml_path = "/data/gauss/lyh/datasets/power_networks/test/xml"
    output_dir = "/data/gauss/lyh/datasets/power_networks/test/annotations.json"
    images = []
    categories = []
    annotations = []

    cat_dict = get_catlist(txt_path)
    for key, value in cat_dict.items():
        categories.append({
            "id": int(value[1]),
            "name": value[0],
            "supercategory": "none"
        })
    # categories.append({
    #     "id": 1,
    #     "name": "objects",
    #     "supercategory": "none"
    # })
    images, annotations = get_images(xml_path, cat_dict)
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

    with open(output_dir, 'w') as jf:
        json.dump(json_file, jf, indent=4)
