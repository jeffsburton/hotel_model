
import torch.nn as nn
from torchvision import transforms, models

import torch.nn as nn
from torchvision import models


class RoomClassifier(nn.Module):
    def __init__(self, num_classes=133):
        super(RoomClassifier, self).__init__()

        # Use ResNet50 with pretrained weights
        self.backbone = models.resnet50(pretrained=True)

        # Freeze early layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # Modify the classifier head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU() ,
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def makeParameters(model):
    parameters = [
        {'params': model.backbone.fc.parameters(), 'lr': 1e-3},
        {'params': model.backbone.layer4.parameters(), 'lr': 1e-4},  # Last layer
        {'params': model.backbone.layer3.parameters(), 'lr': 1e-5},  # Second to last layer
        {'params': [p for n, p in model.backbone.named_parameters()
                    if not any(l in n for l in ['fc', 'layer3', 'layer4'])],
         'lr': 0}  # Earlier layers frozen
    ]
    return parameters

def get_num_classes_from_checkpoint(checkpoint):
    """Determine number of classes from model checkpoint"""
    # The final layer weights will have shape [num_classes, feature_dim]
    # We can find this by looking for the largest weight tensor ending with '.weight'
    # in the backbone.fc layers
    num_classes = None
    for key, value in checkpoint['model_state_dict'].items():
        if 'backbone.fc' in key and key.endswith('.weight'):
            num_classes = value.size(0)

    if num_classes is None:
        raise ValueError("Could not determine number of classes from checkpoint")

    return num_classes



""""
class RoomClassifier(nn.Module):
    def __init__(self, num_classes=133):
        super(RoomClassifier, self).__init__()

        # Use EfficientNet-B0 with pretrained weights
        self.backbone = models.efficientnet_b3(pretrained=True)

        # Unfreeze only the later layers initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers (adjust as needed)
        for param in self.backbone.features[-4:].parameters():
            param.requires_grad = True

        # Modified classifier head
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def makeParameters(model):

    # Adjusted learning rates for small dataset
    parameters = [
        {'params': model.backbone.classifier.parameters(), 'lr': 1e-3},
        {'params': model.backbone.features[-3:].parameters(), 'lr': 1e-4},  # Last 3 layers
        {'params': model.backbone.features[:-3].parameters(), 'lr': 0},  # Earlier layers frozen
        {'params': [p for n, p in model.backbone.named_parameters()
                    if not any(l in n for l in ['classifier', 'features'])],
         'lr': 0}  # Other parameters frozen
    ]
    return parameters
"""