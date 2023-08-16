from .train.torchvision.models import get_model
import torch.nn as nn

def EfficientNet(arch,num_classes):
    model = get_model("efficientnet_v2_" + arch, weights="IMAGENET1K_V1")
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2,inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    )
    return model