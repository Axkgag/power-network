import torch
from torchvision import transforms
from torch.nn import functional as F

import numpy as np
import os
import csv

from read_json import Config
config = Config("config.json")
cuda = config.use_cuda
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus

# Set up data loaders
from datasets import CompDataset
from datasets import SiameseComp

# Set up the network and training parameters
from networks_for_focal import SiameseNet

pos_ids = list(config.pos_map.keys())

def test(val_loader, model, cuda):
    confusion_matrix = np.zeros((2, 2), dtype=np.int32)
    confusion_matrix_by_pos = {p: np.zeros((2, 2), dtype=np.int32) for p in pos_ids}
    with open(config.test_results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['pos', 'img', 'img_name', 'target', 'prediction', 'score'])
        with torch.no_grad():
            model.eval()
            for batch_idx, (data, target, img2_name, pos) in enumerate(val_loader):
                print(str(batch_idx) + '\r', end="")
                imgs1 = data[0]
                img2 = data[1]
                predictions = []
                scores = 0
                for img1 in imgs1:
                    inp = [img1, img2]
                    if cuda:
                        inp = (tuple(d.cuda() for d in inp),)
                    else:
                        inp = (tuple(d for d in inp),)
                    outputs = model(*inp, is_training=False)
                    binary_scores = outputs[0]
                    binary_scores = torch.nn.Softmax(dim=0)(binary_scores)
                    prediction = binary_scores[1] > 0.5
                    scores += binary_scores[1].cpu().numpy()
                    predictions.append(prediction)
                
                predictions = np.array(predictions)
                if len(imgs1)>0:
                    scores /= len(imgs1)
                pred_ok = (predictions == 1).sum() > (predictions == 0).sum()
                gt = target.numpy()[0]
                pos = pos[0]
                if not pos in confusion_matrix_by_pos:
                    confusion_matrix_by_pos[pos] =np.zeros((2, 2))
                if gt and pred_ok:
                    confusion_matrix[0, 0] += 1
                    confusion_matrix_by_pos[pos][0, 0] += 1
                elif gt and (not pred_ok):
                    confusion_matrix[0, 1] += 1
                    confusion_matrix_by_pos[pos][0, 1] += 1
                elif (not gt) and pred_ok:
                    confusion_matrix[1, 0] += 1
                    confusion_matrix_by_pos[pos][1, 0] += 1
                    print(pos, img2_name[0])
                else:
                    confusion_matrix[1, 1] += 1
                    confusion_matrix_by_pos[pos][1, 1] += 1
                writer.writerow([pos, img2_name[0], os.path.basename(img2_name[0]), 1-int(gt), 1-int(pred_ok), scores])
    
    print(confusion_matrix)
    for k, v in confusion_matrix_by_pos.items():
        print(k)
        print(str(v[0, 0]) + '\t' + str(v[0, 1]) + '\n' + str(v[1, 0]) + '\t' + str(v[1, 1]))


if __name__ == "__main__":
    print("begin test")
    test_dataset = CompDataset(config.test_file, anchor_path=config.train_file, train=False,
                                transform=transforms.Compose([
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]))

    siamese_test_dataset = SiameseComp(test_dataset)

    batch_size = 1
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    model = SiameseNet(config.cls_model_name)
    weights_name = config.model_prefix + str(config.epochs -1) + ".pkl"
    if cuda:
        state_dict = torch.load(weights_name)
        model.load_state_dict(state_dict)
        model.cuda()
    else:
        state_dict = torch.load(weights_name, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    test(siamese_test_loader, model, cuda)
    print("finish test")

