import torch

from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

import time
import os

def get_datasets(data_dir, input_size=224, batch_size=128, num_workers=16):
    # input_size = 224
    # batch_size = 128

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # data_dir = './data/'
    image_datasets = {
        x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']
    }

    # Create training and validation dataloaders
    data_loaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                    shuffle=True if x == 'train' else False,
                                    num_workers=num_workers) for x in ['train', 'test']
    }

    idx_to_class = {v: k for k, v in image_datasets['train'].class_to_idx.items()}

    return data_loaders, image_datasets, idx_to_class

if __name__ == "__main__":
    train, test, idx_to_class = get_datasets("./data/", batch_size=1)
    labels_dist = {}
    for inputs, labels in train:
        # print(inputs.shape, idx_to_class[labels])
        cls = labels[0]
        if cls in labels_dist:
            labels_dist[cls] += 1
        else:
            labels_dist[cls] = 1

    print(labels_dist)
