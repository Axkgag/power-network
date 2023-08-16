# RCenterNet算法参数字段：

## 训练参数字段：

|        字段         |           类型           |             解释             |                默认值                 |           取值范围           | 用户是否可更改 |
|:-----------------:|:----------------------:|:--------------------------:|:----------------------------------:|:------------------------:|:-------:|
|      "arch"       |          str           |          预训练网络架构           |              "res_50"              |      res_18, res_50      |    是    |
|   "batch_size"    |          int           |            批大小             |                 8                  |            >0            |    是    |
|   "class_names"   |  list[str, str, ...]   |           数据类别名            |            ["1","2",..]            |                          |    是    |
|   "export_dir"    |          str           |           结果输出路径           |   "../../ChipCounter/data/chip/"   |                          |    是    |
|    "input_res"    |     list[int, int]     |          输入网络图片大小          |             [512,512]              |                          |    否    |
|       "lr"        |         float          |            学习率             |               0.001                |                          |    否    |
|     "lr_step"     |     "float, float"     |           学习率下降            |             "0.7, 0.9"             |                          |    否    |
| "score_threshold" | list[float, float,...] |           置信度阈值            |           [0.5. 0.5,...]           | 取值范围(0,1)，列表中元素个数需与类别数一致 |    是    |
|     "iou_thr"     |         float          |           iou阈值            |                0.4                 |          (0,1)           |    是    |
|   "num_classes"   |          int           |           数据类别数            |                 13                 |         与类别名个数相同         |    是    |
|   "num_epochs"    |          int           |            训练轮数            |                150                 |            >0            |    是    |
|   "num_workers"   |          int           |          读取数据线程数           |                 0                  |           >=0            |    是    |
|      "gpus"       |          str           | gpu推理(-1表示使用cpu，其他表示指定gpu) |                "0"                 |     "-1","0","1"...      |    是    |
|      "mean"       | list[float,float,...]  |                            | [0.40789654,0.44719302,0.47026115] |                          |    否    |
|       "std"       | list[float,float,...]  |                            | [0.28863828,0.27408164,0.27809835] |                          |    否    |
| "target_program"  |          str           |                      |               "BXS"                |                          |    否    |
|      "task"       |          str           |                      |              "ctdet"               |                          |    否    |
|  "val_intervals"  |          int           |     <0不进行验证，>0每n轮验证一次      |                 -1                 |          -1, >0          |    是    |
|   "patch_train"   |          int           |          是否采用分块训练          |                 0                  |           0,1            |    是    |
|     "crop_w"      |          int           |      分块后图片宽度（不分块则无此项）      |                1024                |        大于0，小于图片宽度        |    是    |
|     "crop_h"      |          int           |      分块后图片高度（不分块则无此项）      |                1024                |        大于0，小于图片宽度        |    是    |
|    "stride_x"     |          int           |     分块时x方向步长（不分块则无此项）      |                768                 |       大于0，小于crop_w       |    是    |
|    "stride_y"     |          int           |     分块时y方向步长（不分块则无此项）      |                768                 |       大于0，小于crop_h       |    是    |


## 测试参数字段：

|        字段         |          类型           |             解释             |     默认值      |      取值范围       | 用户是否可更改 |
|:-----------------:|:---------------------:|:--------------------------:|:------------:|:---------------:|:-------:|
|      "arch"       |          str          |          预训练网络架构           |   "res_50"   | res_18, res_50  |    是    |
|   "class_names"   |  list[str, str, ...]  |           数据类别名            | ["1","2",..] |                 |    是    |
|    "input_res"    |    list[int, int]     |          输入网络图片大小          |  [512,512]   |                 |    否    |
| "score_threshold" |         float         |           置信度阈值            |     0.5      |      (0,1)      |    是    |
|     "iou_thr"     |         float         |           iou阈值            |     0.4      |      (0,1)      |    是    |
|   "num_classes"   |          int          |           数据类别数            |      13      |    与类别名个数相同     |    是    |
|      "fp16"       |         bool          |           半精度推理            |    false     |   true,false    |    是    |
|      "gpus"       |          str          | gpu推理(-1表示使用cpu，其他表示指定gpu) |     "0"      | "-1","0","1"... |    是    |
|        "K"        |          int          |           最大目标个数           |     100      |       大于0       |    是    |
|      "mean"       | list[float,float,...] |                            |     [0.40789654,0.44719302,0.47026115]      |             |    否    |
|       "std"       | list[float,float,...] |                            |     [0.28863828,0.27408164,0.27809835]      |             |    否    |
|     "crop_w"      |          int          |      分块后图片宽度（不分块则无此项）      |     1024     |   大于0，小于图片宽度    |    是    |
|     "crop_h"      |          int          |      分块后图片高度（不分块则无此项）      |     1024     |   大于0，小于图片宽度    |    是    |
|    "stride_x"     |          int          |     分块时x方向步长（不分块则无此项）      |     768      |  大于0，小于crop_w   |    是    |
|    "stride_y"     |          int          |     分块时y方向步长（不分块则无此项）      |     768      |  大于0，小于crop_h   |    是    |



# RCenterNet 算法目录结构：

- ${train_dir}
  - cache/
  - image/
  - trainer.json
  - annotations.json

- ${export_dir}
  - model_best.pth
  - model_best.onnx
  - predictor.json
    
