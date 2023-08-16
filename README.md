## 目录
1. [生成分块数据以及标签]()
2. [滑动分块预测]()




## 1.生成分块数据以及标签

### (1). 代码路径：riemann_interface/defaults.py
#### 新增函数 cvtColor()， imageConversion()， setupDataset()，  cvtColor()和imageConversion()在setupDataset()中被调用。
```python
class AbstractTrainer:
    def __init__(self,...):
        pass
    
    def cvtColor(self, ...):
        pass
    def imageConversion(self,...):
        pass
    def setupDataset(self,...):
        pass

```



### (2). setupDataset()函数功能：

```python
def setupDataset(self, train_annotation_path: str, train_img_dir: str, crop_image_dir: str, crop_json_path: str):
    """对COCO格式数据集中的图片进行crop处理，产生新的数据和标签，并保存到默认位置（模型训练时使用新产生的数据集训练）。
       输入参数自定义。
    :train_annotation_path:  crop之前coco数据的.json文件路径
    :train_img_dir:          crop之前数据图片的路径
    :crop_image_dir:         crop后图片存放的文件夹，提前定义好
    :crop_json_path:         crop之后生成coco数据的.json文件路径，提前定义好
    """

```

### (3). setupDataset()调用示例：
#### 根据trainer.json内的patch_train字段，判断是否使用分块训练
#### 使用分块训练，则调用setupDataset()，并更改数据路径
```python
if train_config["patch_train"]:
    self.crop_w = train_config["crop_w"]
    self.crop_h = train_config["crop_h"]
    self.stride_x = train_config["stride_x"]
    self.stride_y = train_config["stride_y"]

    crop_train_img_path = os.path.join(train_dir, "crop_image")
    crop_train_anno_path = os.path.join(train_dir, "crop_annotations.json")
    
    self.setupDataset(train_anno_path, train_img_path, crop_train_img_path, crop_train_anno_path)

    crop_val_img_path = os.path.join(train_dir, "crop_val")
    crop_val_anno_path = os.path.join(train_dir, "crop_val_annotations.json")
    
    self.setupDataset(val_anno_path, val_img_path, crop_val_img_path, crop_val_anno_path)

    self.exp.train_ann = "crop_annotations.json"
    self.exp.train_img = "crop_image"
    self.exp.val_ann = "crop_val_annotations.json"
    self.exp.val_img = "crop_val"
```

### (4). 示例代码：gen_crop_data.py


```python
"""
生成数据滑动crop之后的训练数据和标签
"""
import os
from riemann_interface.defaults import AbstractTrainer
from utils.logsender import LogSender

mysender = LogSender()
mode = 'Rotation_Detection'          # 检测任务模式, 'Detection' or 'Rotation_Detection', 默认为'Rotation_Detection'

train_annotation_path = r"D:\task\ChipCounter\data\chip\before_annotations.json"  # crop之前coco数据的.json文件
train_img_dir = r"D:\task\ChipCounter\data\chip\before_image"    # crop之前的coco数据对应的图片文件夹
crop_image_dir = r"D:\task\ChipCounter\data\chip\image"  # crop之后生成图片的文件夹，(路径必须存在)
crop_json_path = r"D:\task\ChipCounter\data\chip\annotations.json"  # crop之后生成coco数据的.json文件(路径可以不存在)

trainer = AbstractTrainer(mysender)
trainer.setupDataset(train_annotation_path=train_annotation_path, 
                     train_img_dir=train_img_dir, 
                     crop_image_dir=crop_image_dir, 
                     crop_json_path=crop_json_path)

```





## 2.滑动分块预测

### (1). 代码路径：riemann_interface/defaults.py

#### 在基类中新增函数 filter_by_iou()， filter_by_iou_plus()， patchPredict()，  filter_by_iou()和filter_by_iou_plus()在patchPredict()中被调用。


```python
class AbstractPredictor():
    def __init__(self,...):
        pass
    
    def filter_by_iou(self, ...):
        raise NotImplementedError

    def filter_by_iou_plus(self, ...):
        raise NotImplementedError

class AbstractObjectDetectPredictor(AbstractPredictor):
    """
    目标检测推理基类
    """
    def setCropConfig(self, config: Dict) -> bool:
        key_list = ["crop_w", "crop_h", "stride_x", "stride_y"]
        for key in key_list:
            if (not key in config) or not isinstance(config[key], int):
                return False
        self.crop_w = config["crop_w"]
        self.crop_h = config["crop_h"]
        self.stride_x = config["stride_x"]
        self.stride_y = config["stride_y"]
        return True
    def setupPredictor(self, config: Dict[str, Any]):
        ...
        self.use_patch = self.setCropConfig(config)
        return True
    def patchPredict(self, img: Any, info: Dict[str, Any] = None): 
        ...
        return ret
    
    def smartPredict(self, img: Any, info: Dict[str, Any] = None):
        if self.use_patch:
            return self.patchPredict(img, info)
        else:
            return self.predict(img, info)

```



### (2). patchPredict()函数功能：

```python
def patchPredict(self, img: Any, info: Dict[str, Any]=None):
    """输入一张图像img，实现分块预测，并将检测结果合并。
    :img:          图片或图片路径，在函数内判断
    """

```



###  (3). filter_by_iou()，filter_by_iou_plus()函数功能:

#### 坐标汇总后，需要Bbox过滤，任务为旋转目标检测的时候，采用filter_by_iou()函数过滤，任务为不带旋转的目标检测时，采用filter_by_iou_plus()函数
```python
if mode == 'Rotation_Detection':
    results = self.filter_by_iou(new_results)
else:
    results = self.filter_by_iou_plus(new_results)
```


#### **注意：** filter_by_iou()没有变化，filter_by_iou_plus()是重新定义的一个nms，它只将filter_by_iou()中的iou公式中的分母改为了min( area(box1),area(box2) )，具体修改如下：


```python
"""filter_by_iou"""
ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)
"""filter_by_iou_plus"""
ious = overlaps / np.minimum(areas[i], areas[index[1:]])
```