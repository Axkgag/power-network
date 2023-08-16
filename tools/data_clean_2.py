import os
import cv2
from PIL import Image
import numpy as np

data_dir = "../../datasets/train_collect/speed/image"

img_list = os.listdir(data_dir)
for img_name in img_list:
    img_path = os.path.join(data_dir, img_name)
    # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    # if img.shape[-1] == 4:
    #     print(img_name)
    img = Image.open(img_path)
    # print(img.format, img.size, img.mode)
    if img.mode == "RGBA":
        print(img_name)
        img = img.convert("RGB")
        # img.save(img_path)
