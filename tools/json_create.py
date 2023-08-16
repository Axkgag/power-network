import json
import xml.etree.ElementTree as ET
import os


img_path = "../../docker_test/input_picture"
json_dir = "../../docker_test/attach/pictureName.json"
img_list = os.listdir(img_path)
img_list.sort()
pictureName = []
for img_id, img_name in enumerate(img_list):
    # tree = ET.parse(os.path.join(img_path, img_name))
    # origin_img_name = tree.find("filename").text
    # img_name = str(img_id) + ".jpg"
    pictureName.append({
        "fileName": img_name,
        "originFileName": img_name
    })

with open(json_dir, 'w') as f:
    json.dump(pictureName, f, indent=4)
