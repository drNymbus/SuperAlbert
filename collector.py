import torch

from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

import time
import os

def normalization_parameter(dataloader):
    mean = 0.
    std = 0.
    nb_samples = len(dataloader.dataset)
    for _, data in enumerate(dataloader):
        data, _ = data

        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= nb_samples
    std /= nb_samples
    return mean.numpy(),std.numpy()

def get_datasets(data_dir, input_size=224, batch_size=128, num_workers=16, device="cpu"):
    # input_size = 224
    # batch_size = 128

    image_datasets = {
        x: ImageFolder(os.path.join(data_dir, x), transforms.Compose([transforms.ToTensor()])) for x in ['train', 'test']
    }

    # Create training and validation dataloaders
    data_loaders = {
        x: torch.utils.data.DataLoader(image_datasets[x]) for x in ['train', 'test']
    }

    # APPLY NORMALIZATION (CENTER & REDUCED)
    mean, std = normalization_parameter(data_loaders["train"])
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    image_datasets = {
        x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']
    }

    # Create training and validation dataloaders
    data_loaders = {
        x: torch.utils.data.DataLoader(image_datasets[x]) for x in ['train', 'test']
    }

    # APPLY NORMALIZATION (CENTER & REDUCED)
    # mean, std = normalization_parameter(data_loaders["train"])

    # transformer = transforms.Compose([transforms.Normalize(mean, std)])
    # images_norm = { x : ImageFolder(image_datasets[x], transformer) for x in ["train", "test"] }
    # data_loaders = {
    #     x : torch.utils.data.DataLoader(images_norm[x]) for x in ["train", "test"]
    # }

    idx_to_class = {v: k for k, v in image_datasets['train'].class_to_idx.items()}

    return data_loaders, image_datasets, idx_to_class

if __name__ == "__main__":
    train, test, idx_to_class = get_datasets("../data_testing/", batch_size=1)
    # labels_dist = {}
    # for inputs, labels in train:
    #     # print(inputs.shape, idx_to_class[labels])
    #     cls = labels[0]
    #     if cls in labels_dist:
    #         labels_dist[cls] += 1
    #     else:
    #         labels_dist[cls] = 1

    print(train)
