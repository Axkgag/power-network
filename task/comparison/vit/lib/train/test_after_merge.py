import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle5 as pickle
import time
import os
import csv
import json


from .networks_for_focal import SiameseNet
from .aug import maker_border_resize
import copy
np.random.seed(666)

def get_model(weights_name=None, cuda=True, model_name="cls18"):
    model = SiameseNet(model_name)
    if weights_name is not None:
        if cuda:
            state_dict = torch.load(weights_name)
            model.load_state_dict(state_dict)
            model.cuda()
        else:
            state_dict = torch.load(weights_name, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
    return model

def get_anchor_embeddings(anchor_path, model, transform, cuda):
    anchor = np.loadtxt(anchor_path, dtype=str, comments=';', encoding='utf-8')
    anchor_imgs = {i:[] for i in pos_ids} # 只保存每个点位的正样本
    with torch.no_grad():
        model.eval()
        for i in range(len(anchor[:, 0])):
            pos = anchor[i, 0].split('#')[-1].replace('.jpg', '').replace('.png', '')
            if not pos in anchor_imgs:
                anchor_imgs[pos]=[]
            if len(anchor_imgs[pos]) > 10: continue
            label = int(anchor[i, 1]) # 获取标签，如果是ok，则添加进anchor_imgs
            if label == 0:
                img1_name = anchor[i, 0]
                #print(img1_name)
                img1 = Image.open(os.path.join(os.path.dirname(anchor_path) , img1_name)).convert("RGB")
                if pos in not_keep_ratio_part:
                    img1 = img1.resize((256, 256))
                else:
                    img1 = maker_border_resize(np.asarray(img1), 256)
                    img1 = Image.fromarray(img1)
                img1 = transform(img1)
                img1 = img1.unsqueeze(0)
                if cuda:
                    img1 = img1.cuda()
                emb = model.get_embedding(img1)
                emb_cpu = emb.cpu()
                anchor_imgs[pos].append(emb_cpu)
    return anchor_imgs

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=4)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def testData(model,img1_embeddings,test_file,print_prefix,save_file):
    
    confusion_matrix = np.zeros((2, 2), dtype=np.int32)
    confusion_matrix_by_pos = {p: np.zeros((2, 2), dtype=np.int32) for p in pos_ids}
    data_folder=os.path.dirname(test_file)
    with torch.no_grad():
        model.eval()
        test_data = np.loadtxt(test_file, dtype=str, comments=';', encoding='utf-8')
        img_paths = test_data[:, 0]
        labels = test_data[:, 1]
        with open(save_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['pos', 'img', 'img_name', 'target', 'prediction', 'score'])
            for i, img_path in enumerate(img_paths):
                print(str(i) + '\r', end="")
                pos = img_path.split('#')[-1].replace('.jpg', '').replace('.png', '')
                if pos not in pos_ids: continue

                predictions = []
                imgs1_embedding = img1_embeddings[pos][:5]
                img2 = Image.open(os.path.join(data_folder, img_path)).convert("RGB")
                if pos in not_keep_ratio_part:
                    img2 = img2.resize((256, 256))
                else:
                    img2 = maker_border_resize(np.asarray(img2), 256)
                    img2 = Image.fromarray(img2)
                img2 = transform(img2)
                img2 = img2.unsqueeze(0)
                if cuda:
                    img2 = img2.cuda()
                img2_embedding = model.get_embedding(img2)
                scores = 0
                for img1_embedding in imgs1_embedding:
                    outputs = model.get_fc(img1_embedding, img2_embedding)
                    binary_scores = outputs[0]
                    binary_scores = torch.nn.Softmax(dim=0)(binary_scores).cpu()
                    prediction = binary_scores[1] > 0.5
                    scores += binary_scores[1].numpy()
                    predictions.append(prediction)
                
                predictions = np.array(predictions)
                if (len(imgs1_embedding)>0) :
                    scores /= len(imgs1_embedding)
                pred_ok = (predictions == 1).sum() > (predictions == 0).sum() 

                gt = (int(labels[i]) == 0)
                #print(pos, gt, scores, len(imgs1_embedding))
                if not pos in confusion_matrix_by_pos:
                    confusion_matrix_by_pos[pos] = np.zeros((2, 2), dtype=np.int32) 
                if gt and pred_ok:
                    confusion_matrix[0, 0] += 1
                    confusion_matrix_by_pos[pos][0, 0] += 1
                elif gt and (not pred_ok):
                    confusion_matrix[0, 1] += 1
                    confusion_matrix_by_pos[pos][0, 1] += 1
                elif (not gt) and pred_ok:
                    confusion_matrix[1, 0] += 1
                    confusion_matrix_by_pos[pos][1, 0] += 1
                    print(pos, img_path)
                else:
                    confusion_matrix[1, 1] += 1
                    confusion_matrix_by_pos[pos][1, 1] += 1
                writer.writerow([pos, img_path, os.path.basename(img_path), 1- int(gt), 1- int(pred_ok), scores])
        print(print_prefix,confusion_matrix)
        for k, v in confusion_matrix_by_pos.items():
            print(k)
            print(str(v[0, 0]) + '\t' + str(v[0, 1]) + '\n' + str(v[1, 0]) + '\t' + str(v[1, 1]))
    return {"ok2ok":int(confusion_matrix[0,0]),
            "ok2ng":int(confusion_matrix[0,1]),
            "ng2ok":int(confusion_matrix[1,0]),
            "ng2ng":int(confusion_matrix[1,1])}

def testAfterMerge(config):
    global cuda,not_keep_ratio_part,pos_ids,transform
    print("begin test no bn")
    
    cuda = config.use_cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    not_keep_ratio_part = config.not_keep_ratio_part
    pos_ids = list(config.pos_map.keys())
    
    model = get_model(weights_name=config.model_nobn_path, cuda=cuda,model_name=config.cls_model_name)
    #print(model)
    transform=transforms.Compose([
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    # 预加载anchor
    
    img1_embeddings = get_anchor_embeddings(config.train_file, model, transform, cuda)
    
    save_obj(img1_embeddings, config.std_embd_path)
    img1_embeddings = load_obj(config.std_embd_path)
    
    
    if cuda:
        for k, v in img1_embeddings.items():
            img1_embeddings[k] = [d.cuda() for d in v]
    print("finish test no bn")       
    return testData(model,img1_embeddings,config.train_file,"train conf:",config.train_result_path)
    

if __name__ == "__main__":
    config={}
    with open("config.json",'r') as f:
        config=json.load(f)
        f.close()
    testAfterMerge(config)

