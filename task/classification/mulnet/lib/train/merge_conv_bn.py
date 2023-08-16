import json
from typing import Dict
import torch
import torch.nn as nn

from .networks_for_focal import EmbeddingNet, SiameseNet


def merge_bn(model):
    ''' merge all 'Conv+BN' (or 'TConv+BN') into 'Conv' (or 'TConv')
    based on https://github.com/pytorch/pytorch/pull/901
    by Kai Zhang (cskaizhang@gmail.com) 
    https://github.com/cszn/DnCNN
    01/01/2019
    '''
    prev_m = None
    for k, m in list(model.named_children()):
        if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)) and (isinstance(prev_m, nn.Conv2d) or isinstance(prev_m, nn.Linear) or isinstance(prev_m, nn.ConvTranspose2d)):

            w = prev_m.weight.data

            if prev_m.bias is None:
                zeros = torch.Tensor(prev_m.out_channels).zero_().type(w.type())
                prev_m.bias = nn.Parameter(zeros)
            b = prev_m.bias.data

            invstd = m.running_var.clone().add_(m.eps).pow_(-0.5)
            if isinstance(prev_m, nn.ConvTranspose2d):
                w.mul_(invstd.view(1, w.size(1), 1, 1).expand_as(w))
            else:
                w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w))
            b.add_(-m.running_mean).mul_(invstd)
            if m.affine:
                if isinstance(prev_m, nn.ConvTranspose2d):
                    w.mul_(m.weight.data.view(1, w.size(1), 1, 1).expand_as(w))
                else:
                    w.mul_(m.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
                b.mul_(m.weight.data).add_(m.bias.data)

            del model._modules[k]
        prev_m = m
        merge_bn(m)


def tidy_sequential(model):
    for k, m in list(model.named_children()):
        if isinstance(m, nn.Sequential):
            if m.__len__() == 1:
                model._modules[k] = m.__getitem__(0)
        tidy_sequential(m)

def merge_conv_bn(config:Dict)->None:
    print("begin convert to no bn")
    model = SiameseNet(config.cls_model_name, config.num_classes)
    weights_name = config.model_bn_path
    state_dict = torch.load(weights_name, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    import os
    os.remove(weights_name)
    # tidy_sequential(model)
    merge_bn(model)
    #print(model)
    torch.save(model.state_dict(), config.model_nobn_path,_use_new_zipfile_serialization=False)
    print("finish convert to no bn")
    
def merge_bn_cpu(path,checkpoint,config)->None:

    model = SiameseNet(config.cls_model_name, config.num_classes)
    state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))["state_dict"]
    model.load_state_dict(state_dict)
    merge_bn(model)
    torch.save(model.state_dict(), path,_use_new_zipfile_serialization=False)

if __name__ == '__main__':
    config={}
    with open("config.json",'r') as f:
        config=json.load(f)
        f.close()
    merge_conv_bn(config)