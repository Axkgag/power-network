import os
import random
import shutil

data_dir = "../../datasets/train_collect/train/cls_train_4/others"
new_data_dir = "../../datasets/train_collect/train/cls_train_4/del"

file_list = os.listdir(data_dir)
for file_name in file_list:
    file_dir = os.path.join(data_dir, file_name)
    new_dir = os.path.join(new_data_dir, file_name)
    if random.random() <= 0.1:
        shutil.move(file_dir, new_dir)