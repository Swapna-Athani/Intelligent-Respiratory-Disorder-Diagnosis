"""Pretrained image feature extractor plus a supervised X-ray classification head."""
from pathlib import Path
import torch
from torch import nn

ARCHITECTURE='resnet18-transfer-v1'
PREPROCESSING='xray-rgb-imagenet-resnet18-v1'

class XrayTransferModel(nn.Module):
    def __init__(self,classes,pretrained=False):
        super().__init__()
        from torchvision.models import resnet18,ResNet18_Weights
        if pretrained:
            torch.hub.set_dir(str(Path(__file__).resolve().parent/'data'/'pretrained'))
        self.backbone=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc=nn.Identity()
        self.register_buffer('feature_mean',torch.zeros(512))
        self.register_buffer('feature_scale',torch.ones(512))
        self.classifier=nn.Linear(512,classes)
    def forward(self,x):
        features=self.backbone(x)
        return self.classifier((features-self.feature_mean)/self.feature_scale)

def transfer_tensor(image):
    from torchvision.models import ResNet18_Weights
    return ResNet18_Weights.DEFAULT.transforms()(image.convert('RGB'))
