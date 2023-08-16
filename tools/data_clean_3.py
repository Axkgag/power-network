import os
import xml.etree.ElementTree as ET
import shutil
import random

def get_xml(data_path):
    old_xml_path = os.path.join(data_path, "small_xml")
    old_img_path = os.path.join(data_path, "small_image")
    xml_list = os.listdir(old_xml_path)
    xml_list.sort()
    new_img_path = os.path.join(data_path, "image")
    if not os.path.exists(new_img_path):
        os.mkdir(new_img_path)
    new_xml_path = os.path.join(data_path, "xml")
    if not os.path.exists(new_xml_path):
        os.mkdir(new_xml_path)
    for xml_name in xml_list:
        tree = ET.parse(os.path.join(old_xml_path, xml_name))
        img_name = tree.find("filename").text
        if random.random() <= 0.5:
            old_img = os.path.join(old_img_path, img_name)
            old_xml = os.path.join(old_xml_path, xml_name)
            new_img = os.path.join(new_img_path, img_name)
            new_xml = os.path.join(new_xml_path, xml_name)
            shutil.move(old_xml, new_xml)
            shutil.move(old_img, new_img)
            


if __name__ == "__main__":
    data_path = "/data/gauss/lyh/datasets/power_networks/train"
    get_xml(data_path)
