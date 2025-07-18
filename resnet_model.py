import torch.nn as nn
from torchvision import models


def get_resnet18_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze layer4 + fc
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model
