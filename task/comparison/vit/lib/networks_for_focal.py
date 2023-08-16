import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet as m
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

class SiameseNet(nn.Module):
    def __init__(self, resnet_name):
        super(SiameseNet, self).__init__()
        self.embedding_net = EmbeddingNet(resnet_name)
        self.output_chanel= output_channels[resnet_name]
        self.conv2d_1x1 = nn.Sequential(nn.Conv2d(2*self.output_chanel, 2048, kernel_size=1, stride=1, padding=0))
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.PReLU(),
                                nn.Linear(512, 2))

    def forward(self, xs, is_training=True):
        with torch.no_grad():
            emb1 = self.embedding_net(xs[0])
            emb2 = self.embedding_net(xs[1])

            emb = torch.cat((emb1, emb2), dim=1)
            emb = self.conv2d_1x1(emb)
            emb_avg_pool = self.avgpool(emb)
            emb_avg_pool = emb_avg_pool.view(emb_avg_pool.size(0), -1)
            outputs = self.fc(emb_avg_pool)
            return outputs
    # @profile(precision=2)
    def get_embedding(self, x):
        gc.collect()
        with torch.no_grad():
            return self.embedding_net(x)
    # @profile(precision=2)
    def get_fc(self, emb1, emb2):
        with torch.no_grad():
            emb = torch.cat((emb1, emb2), dim=1)
            emb = self.conv2d_1x1(emb)
            emb_avg_pool = self.avgpool(emb)
            emb_avg_pool = emb_avg_pool.view(emb_avg_pool.size(0), -1)
            outputs = self.fc(emb_avg_pool)
            return outputs

