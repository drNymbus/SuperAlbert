import torch

from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

from timm.data.transforms_factory import create_transform
from timm.data import ImageDataset

import time
import os

# def normalization_parameter(dataloader):
#     mean = 0.
#     std = 0.
#     nb_samples = len(dataloader.dataset)
#     for _, data in enumerate(dataloader):
#         data, _ = data

#         batch_samples = data.size(0)
#         data = data.view(batch_samples, data.size(1), -1)
#         mean += data.mean(2).sum(0)
#         std += data.std(2).sum(0)
#     mean /= nb_samples
#     std /= nb_samples
#     return mean.numpy(),std.numpy()

def get_datasets(data_dir, input_size=224, batch_size=128, num_workers=16):
    # input_size = 224
    # batch_size = 128

    # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406] # imageNet numbers
    # [0.185571362 0.161665897 0.174258414]
    std  = [0.229, 0.224, 0.225] # imageNet numbers

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
        "train": torch.utils.data.DataLoader(image_datasets["train"]),
        "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=batch_size, num_workers=num_workers)
    }

    idx_to_class = {v: k for k, v in image_datasets['train'].class_to_idx.items()}

    return data_loaders, image_datasets, idx_to_class

def get_data_loader(dir, input_size=224, batch_size=128, num_workers=16, device="cpu"):
    dataset = ImageDataset(dir, 
                       transform=create_transform(input_size, is_training=True))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    return loader, dataset#, idx_to_class


if __name__ == "__main__":
    data_loaders, image_datasets, idx_to_class = get_datasets("../data/", batch_size=128)
    # data_loaders, image_datasets, idx_to_class = get_datasets("/home/data/challenge_2022_miashs/", batch_size=128)
    # labels_dist = {}
    # for inputs, labels in train:
    #     # print(inputs.shape, idx_to_class[labels])
    #     cls = labels[0]
    #     if cls in labels_dist:
    #         labels_dist[cls] += 1
    #     else:
    #         labels_dist[cls] = 1
    print(normalization_parameter(data_loaders["train"]))
    # print(train)
