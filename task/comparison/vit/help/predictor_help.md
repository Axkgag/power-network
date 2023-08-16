# 模型文件夹说明
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


# predictor.json 字段说明

```
{
    "resnet_name": "resnet18",  //*权重适应的基架网络 ， 可能项为  resnet18，resnet50，resnet101  根据trainer.json 决定
    "use_cuda": false           //*是否使用GPU 默认为 不使用
}
```
