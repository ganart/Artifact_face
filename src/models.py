import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from transformers import ViTForImageClassification, ViTConfig


class ResNet_Artifact(nn.Module):
    """
    ResNet50 model initialization.

    Args:
    pretrained(bool): Use ImageNet pretrained weights. default: True
    """
    def __init__(self, pretrained=True):
        super(ResNet_Artifact, self).__init__()
        if pretrained:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  
        else:
            self.model = resnet50(weights=None)  
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, 2))  

    def forward(self, x):
        return self.model(x)


class EfficientNet_Artifact(nn.Module):
    """
    EfficientNet_v2_s model initialization.

    Args:
    pretrained(bool): Use ImageNet pretrained weights. default: True
    """
    def __init__(self, pretrained=True):
        super(EfficientNet_Artifact, self).__init__()
        if pretrained:
            self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        else:
            self.model = efficientnet_v2_s(weights=None)

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.classifier[1].in_features, 2)
        )

    def forward(self, x):
        return self.model(x)


class ViT_Artifact(nn.Module):
    """
    Vision Transformer (ViT) for binary image classification.

    ViT model initialization.

    Args:
    pretrained(bool): Use pretrained weights. default: True
    """
    def __init__(self, pretrained=True):
        super(ViT_Artifact, self).__init__()

        if pretrained:
            self.model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224',
                num_labels=2,  #
                ignore_mismatched_sizes=True
            )
        else:
            config = ViTConfig.from_pretrained('google/vit-base-patch16-224', num_labels=2)
            self.model = ViTForImageClassification(config)

    def forward(self, x):

        outputs = self.model(pixel_values=x)
        return outputs.logits
