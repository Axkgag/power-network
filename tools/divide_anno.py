import os
import json
from pycocotools.coco import COCO

if __name__ == "__main__":
    categories = [[1, 2, 6, 7, 12, 14],
                  [3, 4, 5, 8, 9],
                  [10, 11, 13]]

    origin_path = "/data/gauss/lyh/datasets/power_networks/yolox/annotations"
    json_name = "val_annotations"
    origin_json_path = os.path.join(origin_path, json_name + ".json")
    out_dir = "/data/gauss/lyh/datasets/power_networks/yolox/annotations"
    origin_json = COCO(origin_json_path)
    
    for idx, cat_list in enumerate(categories):
        img_list = set()
        for cat in cat_list:
            img_idx = origin_json.getImgIds(catIds=cat)
            img_list |= set(img_idx)
        img_list = list(img_list)
        anno_list = origin_json.getAnnIds(imgIds=img_list, catIds=cat_list)

        imgs = origin_json.loadImgs(ids=img_list)
        annos = origin_json.loadAnns(ids=anno_list)
        cats = origin_json.loadCats(ids=cat_list)

        outputs = {
            "categories": cats,
            "images": imgs,
            "annotations": annos
        }
        
        with open(os.path.join(out_dir, json_name + "_{}.json".format(str(idx + 1))), 'w') as f:
            json.dump(outputs, f, indent=4)

    

