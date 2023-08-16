# MulNet算法参数字段：

## 训练参数字段：字段，默认值，解释

### "batch_size": 4, 训练时批大小（用户可修改）

### "cls_model_name": "resnet18", 使用的模型基架 （用户可修改）

### "num_classes":2，分类类别 （用户可修改）

### "forbiddens": [], 不参与训练的点位 （用户可修改）

### "epochs": 10, 总共训练的轮数 （用户可修改）

### "learning_rate": 0.001, 学习率 （用户可修改）

### "gamma": 0.1, 学习率下降倍率（用户可修改）

### "steps": [0.7, 0.9], 学习率下降时的轮数（用户可修改），按比例值

### "gpus": "0", 指定gpu（用户可修改）

### "log_interval": 100, 每100步记录一次loss信息（用户可修改）


### "export_dir"：默认train_result_dir，保存测试时需要的数据（用户可修改）

### "momentum": 0.9, 动量（不推荐修改）

### "num_workers": 4, 读取数据时的线程数（不推荐修改）

### "pin_memory": false, 锁页内存（不推荐修改）

### "save_interval": 10, 保存模型权重的间隔数（用户可修改）

### "target_program": "temp2",

### "train_file": "C:/Users/liman_003/Downloads/test@2021-09-16/label.txt", 训练标签
### "test_file": "C:/Users/liman_003/Downloads/test@2021-09-16/label.txt",  测试标签
### "val_file": "C:/Users/liman_003/Downloads/test@2021-09-16/label.txt",  验证标签

### "use_cuda": 1, 是否使用cuda （用户可修改）
### "weight_decay": 0.0001，正则化权重 （不推荐修改）


## 测试参数字段：

### "resnet_name": 训练参数字段cls_model_name

### "classes_name": [], 类别名称（需要和训练时类别顺序对应）（用户可修改）

### "use_cuda": true 

# MulNet算法目录结构：

- ${train_dir}
  - cache
    - *.pkl
  -  report
     - train_result.csv
  - trainer.json

- ${export_dir}
  - model_param
    - model-weight.pkl
    - part-resize.json
    - threshold.json
  - standard_data
    - std_embeddings.pkl
  - predictor.json
    
