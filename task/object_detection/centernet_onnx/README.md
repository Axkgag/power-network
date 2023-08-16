# CenterNet算法参数字段：

## 训练参数字段：字段，类型，解释

### img_dir str 图片目录

### train_annot_path str 训练标签路径

### test_annot_path str 测试标签路径

### export_dir str 实验结果输出目录

### class_names list[str, str, ...] 类别名称，以list形式

### num_classes int  类别个数

### input_res list[int, int] 图像输入尺寸

### task str 任务id：这里只支持'ctdet'

### exp_id str 实验任务名：每次实验应都取一个新的

### load_model str 预训练模型的路径

### resume bool 是否从最新的checkpoint开始训练

### gpus str '-1' for cpu, '0, 1' for id 0 and id 1 gpus

### num_workers int，读取数据时启动的线程数 

### arch str 网络架构 dla_34| res_18 | res_101 | resdcn_18 | resdcn_101 |dlav0_34 |  hourglass

### lr float，学习率

### lr_step str '0.7, 0.9'， 在指定epoch数时学习率下降

### num_epochs int， 总共训练轮数

### batch_size int， 批大小

### score_threshold [0.5,0.5] 检测任务得分阈值

### iou_thr 0.4 后处理，用于过滤iou大于iou_thr的bboxes


## 测试参数字段：

### score_threshold [0.5,0.5] 检测任务得分阈值

### iou_thr 0.4 后处理，用于过滤iou大于iou_thr的bboxes

### class_names list[str, str, ...] 类别名称，以list形式

### num_classes int  类别个数

---

---不使用分块检测则无以下四个参数---

### "crop_w": int 裁切宽度

### "crop_h": int 裁切高度

### "stride_x": int 滑窗x方向步长

### "stride_y": int 滑窗y方向步长





# CenterNet_onnx 算法目录结构：

- ${train_dir}
  - cache/
  - image/
  - trainer.json
  - annotations.json

- ${export_dir}
  - model_best.onnx
  - predictor.json
    
