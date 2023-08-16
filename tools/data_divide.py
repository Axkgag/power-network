import os
import shutil
import random


old_path = "../datasets/power_networks/train"
new_path = "../datasets/power_networks/test"

old_img_path = os.path.join(old_path, "image")
old_xml_path = os.path.join(old_path, "xml")
new_img_path = os.path.join(new_path, "image")
new_xml_path = os.path.join(new_path, "xml")


img_list = os.listdir(old_img_path)
xml_list = os.listdir(old_xml_path)
img_list.sort()
xml_list.sort()

for img_name, xml_name in zip(img_list, xml_list):
    if img_name.split(".")[0] != xml_name.split(".")[0]:
        print(img_name, "!=", xml_name)
        continue
    if random.random() <= 0.05:
        old_img = os.path.join(old_img_path, img_name)
        new_img = os.path.join(new_img_path, img_name)
        old_xml = os.path.join(old_xml_path, xml_name)
        new_xml = os.path.join(new_xml_path, xml_name)
        shutil.move(old_img, new_img)
        shutil.move(old_xml, new_xml)

# shutil.move("old", "new")