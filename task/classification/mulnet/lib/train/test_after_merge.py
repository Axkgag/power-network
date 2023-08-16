# cython: language_level=3
import torch
import torch.nn as nn
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image
import numpy as np
import time
import os
import csv
import json


from .networks_for_focal_no_bn import SiameseNet
from .aug import maker_border_resize
import copy
np.random.seed(666)

def get_model(weights_name=None, cuda=True, model_name="cls18", num_classes=2):
    model = SiameseNet(model_name, num_classes)
    if weights_name is not None:
        if cuda:
            state_dict = torch.load(weights_name)
            model.load_state_dict(state_dict)
            model.cuda()
        else:
            state_dict = torch.load(weights_name, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
    return model

def testData(model,test_file,print_prefix,save_file, num_classes):
    
    total_confusion_matrix = np.zeros((num_classes+1, num_classes+1), dtype=np.float32) # 总的混淆矩阵
    with torch.no_grad():
        model.eval()
        test_data = np.loadtxt(test_file, dtype=str, encoding='utf-8', comments=';')
        data_folder=os.path.dirname(test_file)
        img_paths = test_data[:, 0]
        labels = test_data[:, 1]
        with open(save_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['pos', 'img', 'img_name', 'target', 'prediction'])
            for i, img_path in enumerate(img_paths):
                print(str(i) + '\r', end="")
                pos = img_path.split('#')[-1].replace('.jpg', '')
                img2 = Image.open(os.path.join(data_folder, img_path)).convert("RGB")
                img2 = maker_border_resize(np.asarray(img2), 256)
                img2 = Image.fromarray(img2)
                img2 = transform(img2)
                img2 = img2.unsqueeze(0)
                if cuda:
                    img2 = img2.cuda()
                scores = model(img2)[0]
                scores = scores.cpu().detach().numpy()
                pred_cls = np.argmax(scores)

                total_confusion_matrix[int(labels[i]), int(pred_cls)] += 1
                writer.writerow([pos, img_path, os.path.basename(img_path), labels[i], pred_cls])
    total_confusion_matrix[:num_classes, num_classes] = np.diagonal(total_confusion_matrix[0:num_classes, 0:num_classes]) / np.sum(total_confusion_matrix[0:num_classes, 0:num_classes], axis=1)
    total_confusion_matrix[num_classes, :num_classes] = np.diagonal(total_confusion_matrix[0:num_classes, 0:num_classes]) / np.sum(total_confusion_matrix[0:num_classes, 0:num_classes], axis=0)
    total_confusion_matrix[num_classes, num_classes] = np.sum(np.diagonal(total_confusion_matrix[0:num_classes, 0:num_classes])) / np.sum(total_confusion_matrix[0:num_classes, 0:num_classes])
    return total_confusion_matrix

def testAfterMerge(config):
    global cuda,transform
    print("begin test no bn")
    
    cuda = config.use_cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    
    model = get_model(weights_name=config.model_nobn_path, cuda=cuda,model_name=config.cls_model_name, num_classes=config.num_classes)
    #print(model)
    transform=transforms.Compose([
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    print("finish test no bn")       
    return testData(model,config.train_file,"train conf:",config.train_result_path, config.num_classes)
    

if __name__ == "__main__":
    config={}
    with open("config.json",'r') as f:
        config=json.load(f)
        f.close()
    testAfterMerge(config)

