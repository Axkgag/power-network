"""
生成数据滑动crop之后的训练数据和标签
"""
import os

from riemann_interface.defaults import AbstractObjectDetectTrainer

from utils.logsender import LogSender

mysender = LogSender()
mode = 'Rotation_Detection'          # 检测任务模式, 'Detection' or 'Rotation_Detection', 默认为'Rotation_Detection'

# train_annotation_path = r"D:\task\ChipCounter\data\chip\annotations.json"  # crop之前coco数据的.json文件
#
# train_img_dir = r"D:\task\ChipCounter\data\chip\image"    # crop之前的coco数据对应的图片文件夹
#
# crop_image_dir = r"D:\task\ChipCounter\data\chip\crop_image"  # crop之后生成图片的文件夹，(路径必须存在)
# crop_json_path = r"D:\task\ChipCounter\data\chip\crop_annotations.json"  # crop之后生成coco数据的.json文件(路径可以不存在)

# h_r, w_r = 3, 3  # 滑动次数
# stride = 256  # 滑动步长 默认256
#

train_annotation_path = r'D:\task\ChipCounter\data\chip\test\crop\annotations.json'
train_img_dir = r'D:\task\ChipCounter\data\chip\test\crop\image'
crop_image_dir = r'D:\task\ChipCounter\data\chip\test\crop\crop_image'
crop_json_path = r'D:\task\ChipCounter\data\chip\test\crop\crop_annotations.json'


trainer = AbstractObjectDetectTrainer(mysender)
trainer.crop_w = 1280
trainer.crop_h = 1280
trainer.stride_x = 640
trainer.stride_y = 640
trainer.setupDataset(train_annotation_path=train_annotation_path, train_img_dir=train_img_dir,
                     crop_image_dir=crop_image_dir, crop_json_path=crop_json_path)

