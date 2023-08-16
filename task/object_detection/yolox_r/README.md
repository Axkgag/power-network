# YoloxR算法参数字段：

## 训练参数字段

|        字段         |         类型          |        解释         |              默认值               |                取值范围                | 用户是否可更改 |
|:-----------------:|:-------------------:|:-----------------:|:------------------------------:|:----------------------------------:|:-------:|
|      "arch"       |         str         |      预训练网络架构      |           "yolox_s"            | yolox_s, yolox_m, yolox_l, yolox_x |    是    |
|   "batch_size"    |         int         |        批大小        |               8                |                 >0                 |    是    |
|   "class_names"   | list[str, str, ...] |       数据类别名       |         ["1","10",..]          |                                    |    是    |
|    "input_res"    |   list[int, int]    |     输入网络图片大小      |           [640,640]            |                                    |    否    |
|       "lr"        |        float        |        学习率        |            0.00015             |                                    |    否    |
|      "gpus"       |         int         |      使用的gpu       |               0                |              0,1,2...              |    是    |
|   "num_classes"   |         int         |       数据类别数       |               13               |              与类别名个数相同              |    是    |
|    "num_epoch"    |         int         |       训练轮数        |              150               |                 >0                 |    是    |
|   "num_workers"   |         int         |      读取数据线程数      |               0                |                >=0                 |    是    |
|   "export_dir"    |         str         |      结果输出路径       | "../../ChipCounter/data/chip/" |                                    |    是    |
| "score_threshold" |        float        |       置信度阈值       |              0.5               |               (0,1)                |    是    |
|     "iou_thr"     |        float        |       iou阈值       |              0.4               |               (0,1)                |    是    |
|  "val_intervals"  |         int         | <0不进行验证，>0每n轮验证一次 |               -1               |               -1, >0               |    是    |
|   "patch_train"   |         int         |     是否采用分块训练      |               0                |                0,1                 |    是    |
|     "crop_w"      |         int         |      分块后图片宽度      |              1024              |             大于0，小于图片宽度             |    是    |
|     "crop_h"      |         int         |      分块后图片高度      |              1024              |             大于0，小于图片宽度             |    是    |
|    "stride_x"     |         int         |     分块时x方向步长      |              768               |            大于0，小于crop_w            |    是    |
|    "stride_y"     |         int         |     分块时y方向步长      |              768               |            大于0，小于crop_h            |    是    |

## 测试参数字段：

|        字段         |         类型          |        解释        |              默认值               |                取值范围                | 用户是否可更改 |
|:-----------------:|:-------------------:|:----------------:|:------------------------------:|:----------------------------------:|:-------:|
|      "arch"       |         str         |     预训练网络架构      |           "yolox_s"            | yolox_s, yolox_m, yolox_l, yolox_x |    是    |
|   "class_names"   | list[str, str, ...] |      数据类别名       |         ["1","10",..]          |                                    |    是    |
|    "input_res"    |   list[int, int]    |     输入网络图片大小     |           [640,640]            |                                    |    否    |
| "score_threshold" |        float        |      置信度阈值       |              0.5               |               (0,1)                |    是    |
|     "iou_thr"     |        float        |      iou阈值       |              0.4               |               (0,1)                |    是    |
|   "num_classes"   |         int         |      数据类别数       |               13               |              与类别名个数相同              |    是    |
|      "fp16"       |        bool         |          半精度推理           |     false     |             true,false             |    是    |
|      "gpus"       |         str         | gpu推理(-1表示使用cpu，其他表示指定gpu) |      "0"      |          "-1","0","1"...           |    是    |
|     "crop_w"      |         int         | 分块后图片宽度（不分块则无此项） |              1024              |             大于0，小于图片宽度             |    是    |
|     "crop_h"      |         int         |     分块后图片高度（不分块则无此项）      |              1024              |             大于0，小于图片宽度             |    是    |
|    "stride_x"     |         int         |     分块时x方向步长（不分块则无此项）     |              768               |            大于0，小于crop_w            |    是    |
|    "stride_y"     |         int         |     分块时y方向步长（不分块则无此项）     |              768               |            大于0，小于crop_h            |    是    |


# YoloxR算法目录结构：

- ${train_dir}
  - image/
  - val/
  - trainer.json
  - annotations.json
  - val_annotations.json

- ${model_dir}
  - latest_ckpt.pth
  - model.onnx
  - predictor.json

