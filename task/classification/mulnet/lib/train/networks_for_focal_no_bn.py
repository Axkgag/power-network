import torch.nn as nn
from . import resnet_no_bn as m
from .attention_modules import SELayer

output_channels=m.output_chanels

class EmbeddingNet(nn.Module):
    def __init__(self,cls_model_name):
        super(EmbeddingNet, self).__init__()
        model=getattr(m,cls_model_name)
        self.convnet = model(pretrained=True)

    def forward(self, x):
        x = self.convnet(x)
        return x

    def get_embedding(self, x):
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
    def __init__(self, cls_model_name, num_classes):
        super(SiameseNet, self).__init__()
        self.embedding_net = EmbeddingNet(cls_model_name)
        self.output_chanel= output_channels[cls_model_name]
        self.mul_cls_net = MulClsNet(self.output_chanel, num_classes)

    def forward(self, xs, is_training=True):
        emb2 = self.embedding_net(xs)

        scores = self.mul_cls_net(emb2)
        return scores

    def get_embedding(self, x):
        return self.embedding_net(x)
    
    def get_fc(self, emb2):
        scores = self.mul_cls_net(emb2)
        return scores

