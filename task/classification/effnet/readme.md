# EffNet算法参数字段：

## 训练参数字段：字段，默认值，解释

### "batch_size": 4, 训练时批大小（用户可修改）

### "arch": "s", 使用的模型基架 （用户可修改）

### "num_classes":2，分类类别 （用户可修改）

### "classes_name": ["OK","NG"], 类别名称 （用户可修改）

### "epochs": 10, 总共训练的轮数 （用户可修改）

### "learning_rate": 0.001, 学习率 （用户可修改）

### "gpus": "0", 指定gpu（用户可修改）

### "log_interval": 100, 每100步记录一次loss信息（用户可修改）

### "export_dir"：默认train_result_dir，保存测试时需要的数据（用户可修改）

### "num_workers": 4, 读取数据时的线程数（不推荐修改）

### "pin_memory": false, 锁页内存（不推荐修改）

### "momentum": 0.5, 动量（不推荐修改）
### "use_cuda": 1, 是否使用cuda （用户可修改）
### "weight_decay": 0.0001，正则化权重 （不推荐修改）

## 测试参数字段：

### "arch": 使用的模型基架

### "classes_name": [], 类别名称（需要和训练时类别顺序对应）（用户可修改）

### "use_cuda": true 

# EffNet算法目录结构：

- ${train_dir}
  - cache
    - *.pkl
  - image
    - *.jpg
  - label.txt
  - trainer.json

- ${export_dir}
  - model-weight.pkl
  - effnet.onnx
  - predictor.json
  - report
    - train_result.csv
    
