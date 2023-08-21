import os
import json
from pycocotools.coco import COCO

if __name__ == "__main__":
    categories = [[1, 2, 3, 4],
                  [5, 6, 7, 8, 9],
                  [10, 11, 12, 13,14]]

    origin_json_path = "./tools/val_annotations.json"
    origin_json = COCO(origin_json_path)
    
    for cat_list in categories:
        img_list = origin_json.getImgIds(catIds=cat_list)
        anno_list = origin_json.getAnnIds(imgIds=img_list, catIds=cat_list)

        imgs = origin_json.loadImgs(ids=img_list)
        annos = origin_json.loadAnns(ids=anno_list)
        cats = origin_json.loadCats(ids=cat_list)


    

