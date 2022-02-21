import torch
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import time
import os


def get_datasets(data_dir, input_size=224, batch_size=128, num_workers=16):

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
    
    image_datasets = {
        x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']
    }

    # Create training and validation dataloaders
    data_loaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], 
                                       batch_size=batch_size,
                                       shuffle=True if x == 'train' else False,
                                       num_workers=num_workers) 
                for x in ['train', 'test']
    }

    #La variable num_workers a un effet sur la vitesse d'apprentissage. Cependant vous
    # etes plusieurs sur la meme machine et vous devez en tenir compte !

    idx_to_class = {v: k for k, v in image_datasets['train'].class_to_idx.items()}

    return data_loaders['train'], data_loaders['test'], idx_to_class




if __name__=='__main__':

    input_size = 224
    batch_size = 128
    data_dir = '../data/challenge_2022_miashs'
    num_workers = 16

    data_train, data_test, idx_to_class = get_datasets(data_dir, input_size, batch_size, num_workers)


