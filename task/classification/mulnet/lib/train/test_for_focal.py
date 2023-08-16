# cython: language_level=3
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
from networks_for_focal import EmbeddingNet, SiameseNet

def test(val_loader, model, cuda):
    with open(config.test_results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['pos', 'img', 'img_name', 'prediction', 'score'])
        with torch.no_grad():
            model.eval()
            for batch_idx, (img2, label2, img2_name, pos) in enumerate(val_loader):
                print(str(batch_idx) + '\r', end="")
                if cuda:
                    img2 = img2.cuda()
                scores = model(img2)[0]
                scores = scores.cpu().detach().numpy()
                pred_cls = int(np.argmax(scores))
                                    
                writer.writerow([pos, img2_name[0], os.path.basename(img2_name[0]), pred_cls, scores[pred_cls]])

if __name__ == "__main__":
    print("begin test")
    test_dataset = CompDataset(config.test_file, train=False,
                                transform=transforms.Compose([
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]))

    siamese_test_dataset = SiameseComp(test_dataset)

    batch_size = 1
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    model = SiameseNet(config.cls_model_name, config.num_classes)
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

