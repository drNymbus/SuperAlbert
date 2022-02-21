import torch
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

def create_model(nb_class, gpu=True):
    # Loading pretrained model
    model = models.resnet18(pretrained=True)

    # setting all parameters as constants
    for p in model.parameters():
        p.requires_grad = False

    # changing the last layer for a newly initialized trainable classifier
    model.fc = nn.Linear(model.fc.in_features, nb_class)

    return model