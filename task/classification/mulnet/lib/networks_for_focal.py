import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet as m
from .attention_modules import SELayer

output_channels=m.output_chanels
# from memory_profiler import profile
import gc
class EmbeddingNet(nn.Module):
    # @profile(precision=2)
    def __init__(self,resnet_name):
        super(EmbeddingNet, self).__init__()
        model= getattr(m,resnet_name)
        self.convnet=model(pretrained=False)

    def forward(self, x):
        with torch.no_grad():
            return self.convnet(x)
            

    def get_embedding(self, x):
        with torch.no_grad():
            return self.forward(x)
    
class MulClsNet(nn.Module):
    def __init__(self, channels, num_classes):
        super(MulClsNet, self).__init__()
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.se_layer = SELayer(channels)
        self.net = nn.Sequential(nn.Linear(channels, 512),
                                nn.PReLU(),
                                nn.Linear(512, num_classes))
    def forward(self, emb):
        emb = self.se_layer(emb)
        emb = self.avgpool(emb)
        emb = emb.view(emb.size(0), -1)
        scores = self.net(emb)
        return scores
    
    def get_embedding(self, emb):
        emb = self.se_layer(emb)
        emb = self.avgpool(emb)
        emb = emb.view(emb.size(0), -1)
        return emb

class SiameseNet(nn.Module):
    def __init__(self, resnet_name, num_classes):
        super(SiameseNet, self).__init__()
        self.embedding_net = EmbeddingNet(resnet_name)
        self.output_chanel= output_channels[resnet_name]
        self.mul_cls_net = MulClsNet(self.output_chanel, num_classes)

    def forward(self, xs, is_training=True):
        with torch.no_grad():
            emb2 = self.embedding_net(xs)

            scores = self.mul_cls_net(emb2)
            return scores

    # @profile(precision=2)
    def get_embedding(self, x):
        gc.collect()
        with torch.no_grad():
            return self.embedding_net(x)
            
    # @profile(precision=2)
    def get_fc(self, emb2):
        with torch.no_grad():
            scores = self.mul_cls_net(emb2)
            return scores

