# 训练文件夹说明
```
--train_dir
    |--trainer.json     *训练配置项
    |--label.txt        *标注文件，格式为 每行2列   imagepath,label， 使用空格分隔  
    |--image            *图片存放根目录 图片名包含PartID信息， 不含格式后缀以 #PartID 结尾 图片格式限制为 jpg 
```

# trainer.json 字段说明
*为必须字段
$为用户应调整字段

```
{
    "batch_size": 32,               //*$batchsize 2的幂次， 推荐值为 显存大小/
    "cls_model_name": "resnet18",   //*$模型基架  可选项为 resnet18 resnet50 resnet101
    "epochs": 20,                   //*$训练轮次    推荐值为
    "excludes": [                   //$不参与训练的点位
    ],
    "export_dir": "/workspace/shared_dir//train_result_dir/OP30-ZYT2005@clsnet@2021-11-04_20-36-09",    //$训练完成后导出到的文件夹
    "forbiddens": [                 //
    ],
    "gamma": 0.1,                   //gamma
    "gpus": "0",                    //$使用的gpu编号
    "learning_rate": 0.001,         //*$学习率
    "log_interval": 100,            //日志间隔（每多少个epoch 输出一次）
    "momentum": 0.9,                //*
    "num_workers": 4,               //*$数据集加载线程数
    "pin_memory": true,             //*锁页内存
    "pos_map": {                    //*所有点位
        "Part-0001": "",            
        "Part-0002": "",
        "Part-0003": "",
        "Part-0004": "",
        "Part-0005": "",
        "Part-0006": "",
        "Part-0007": "",
        "Part-0008": "",
        "Part-0009": "",
        "Part-0010": "",
        "Part-0011": "",
        "Part-0012": "",
        "Part-0013": "",
        "Part-0014": "",
        "Part-0015": ""
    },
    "save_interval": 10,            //*$保存模型间隔
    "steps": [                      //学习率更新轮次
        60,
        100
    ],
    "target_program": "OP30-ZYT2005",   //目标程序名（软件内使用）
    "train_file": "shared_dir/train_dir/{244694d0-5ef6-4f8e-a0a5-acd43fcb17db}/label.txt",  //*$训练文件
    "use_cuda": 1,                  //*$是否使用GPU默认为true
    "weight_decay": 0.0001          //*
}
```



# 模型文件夹说明x
训练完成后导出文件夹
```
--model_dir
    |--predictor.json   *运行时加载配置文件
    |--model_weights    *模型权重文件夹
        |--model-weight.pkl     *主要权重
        |--part-resize.json     需要resize的点位，默认为空
        |--threshold.json       每个点位的 判定ok ng的阈值，默认为每个点位 都是 0.5，可调整
    |--standard_data    *标准数据文件文件夹
        |--std_embeddings.pkl   *标准图保存成的字典，内部数据为  {PartIDL:[img1,...],...} 
    |--report           测试报告文件夹
        |--train_result.csv     训练后测试集上的结果
```   



