import timm

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

# from collections import OrderedDict

# from resnet_model import create_pretrained_resnet

def create_model_b4(output_size):
    # Loading pretrained model
    model = models.efficientnet_b4(pretrained=True)
    # setting all parameters as constants
    for p in model.parameters():
        p.requires_grad = False

    # changing the last layer for a newly initialized trainable classifier
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_size)

    return model

def create_model_b3(output_size):
    # Loading pretrained model
    model = models.efficientnet_b3(pretrained=True)
    # setting all parameters as constants
    for p in model.parameters():
        p.requires_grad = False

    # changing the last layer for a newly initialized trainable classifier
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_size)

    return model

def create_model_resnet(output_size):
    model = timm.create_model('resnet50', pretrained=True, num_classes=output_size)
    return model

def create_model_densenet(output_size):
    model = timm.create_model('densenet201', pretrained=True, num_classes=output_size)
    return model