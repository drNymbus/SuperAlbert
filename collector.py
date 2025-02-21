import torch

from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

from timm.data.transforms_factory import create_transform
from timm.data import ImageDataset

import numpy as np

import time
import os
import json

# import utils

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

def get_dataset(dir, input_size=224):
    folder = ImageFolder(dir, input_size=input_size)
    file = ImageDataset(dir, transform=create_transform(input_size, is_training=True))
    return file, folder

def get_indices_and_classes(dir, input_size=224):
    dataset = ImageFolder(dir)
    # print("dataset(get_indices_and_classes): ", len(dataset))
    idx2cls = {v:k for k, v in dataset.class_to_idx.items()}
    # print("idx: ", len(idx2cls))
    return idx2cls, dataset.class_to_idx

def get_sampler(filename, dataset, idx2cls):
    freq = np.genfromtxt(filename, delimiter=';', dtype='int')
    counts = freq[:,0]
    labels = freq[:,1]
    idx = freq[:,2]

    class_weights = [1/c for c in counts]
    #example_weights = [class_weights[np.where(labels == int(idx2cls[image[1]]))[0][0]] for image in dataset]
    example_weights = [class_weights[image[1]] for image in dataset]
    sampler = torch.utils.data.WeightedRandomSampler(example_weights, len(dataset))

    return sampler

def get_data_loader(dataset, sampler=None, shuffle=False, batch_size=128, num_workers=16, device="cpu"):
    # # sampler = utils.get_sampler()
    # shuffle = shuffle if sampler is None else False
    # print("dataset(get_data_loader): ", len(dataset))
    # # for p in dataset.parser:
    # #     print(p)
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # for b in loader:
    # print("loader(get_data_loader): ", len(loader))
    # idx_to_class = {v: k for k, v in dataset.parser.class_to_idx.items()}

    return loader, dataset#, idx_to_class

def get_dataloader(dir, input_size=224, sampler=None, shuffle=False, batch_size=128, num_workers=16, device="cpu"):
    dataset, folder = get_dataset(dir, input_size=input_size)
    print("dataset")
    idx2cls, _ = get_indices_and_classes(dir, input_size=input_size)
    print("idx2cls")
    sampler = None if sampler is None else get_sampler(sampler, folder, idx2cls)
    print("sampler")
    return get_data_loader(dataset, sampler, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

if __name__ == "__main__":
    trainset, train_img = get_dataloader("../data/train", sampler="./data_aux/frequencies.csv", batch_size=1, num_workers=4)
    print("final length(main): ", len(trainset))
    # print(train)
