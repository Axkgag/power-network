import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def fix_bn(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

class SiameseNet(nn.Module):
    def __init__(self, cls_model_name):
        super(SiameseNet, self).__init__()
        self.embedding_net = timm.create_model(cls_model_name, pretrained=True)
        self.embedding_net.head = nn.Identity()
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.PReLU(),
                                nn.Linear(512, 2))
    
    def freeze_embedding_net(self):
        for param in self.embedding_net.parameters():
            param.requires_grad = False
        self.embedding_net.apply(fix_bn)

    def forward(self, xs, is_training=True):
        emb1 = self.embedding_net(xs[0])
        emb2 = self.embedding_net(xs[1])

        emb = torch.cat((emb1, emb2), dim=1)
        outputs = self.fc(emb)
        return outputs

    def get_embedding(self, x):
        return self.embedding_net(x)
    
    def get_fc(self, emb1, emb2):
        emb = torch.cat((emb1, emb2), dim=1)
        outputs = self.fc(emb)
        return outputs

