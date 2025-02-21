import math
import time
import os
import json

import numpy as np

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

from sklearn.metrics import f1_score, top_k_accuracy_score

import collector

def create_model_dir(name):
    if not os.path.isdir("results/" + name):
        try:
            os.mkdir("results/" + name)
            return "results/" + name + '/'
        except Exception as e:
            print("CANNOT INIT DIR : " + str(e))
            return None
    return "results/" + name + '/'

def get_device():
    # Detect if we have a GPU available
    if torch.cuda.is_available():
        return torch.device("cuda:2")
    return torch.device("cpu")

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    return model

#########################################################
##################  TRAINING  ###########################
#########################################################

def test_model(model, dataset, device=None, ssout=False):
    model.eval()

    predictions = []
    predictions_score = []
    y_true = []

    for _, item in enumerate(dataset):
        inputs, labels = item
        if device is not None:
            inputs = inputs.to(device)
            labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            # Special case for inception because in training it has an auxiliary output. In train
            #   mode we calculate the loss by summing the final output and the auxiliary output
            #   but in testing we only consider the final output.
            outputs = model(inputs)
            outputs_argmax = outputs.argmax(dim=1)
            # print(inputs.shape, outputs.shape)
            for i in range(inputs.shape[0]):
                predictions.append(outputs_argmax.cpu().numpy()[i]) #outputs.argmax(dim=1).cpu().numpy()[0]
                predictions_score.append(list(outputs.cpu().numpy()[i]))
                y_true.append(labels.cpu().numpy()[i])

            if ssout:
                print("({}/{}) Testing ...".format(i+1, len(dataset)), end="\r")
    
    print()

    score = {
        "f_weighted": -1,
        "f_macro": -1,
        "top_k": -1
    }

    try:
        score["f_weighted"] = f1_score(y_true, predictions, average="weighted")
        score["f_macro"] = f1_score(y_true, predictions, average="macro")
        score["top_k"] = top_k_accuracy_score(y_true, predictions_score)
    except Exception as e:
        print("Error compute scoring")
        print(str(e))

    return score

def train_model(model, dataset, validation, criterion, optimizer, decay, batch_size=128, num_epochs=5, num_workers=16, device="cpu", history="results/log.txt", ssout=False):
    since = time.time()
    # history = {}

    with open(history, 'a') as f:
        header = "e, loss"
        if validation:
            header += ", score"
        f.write(header + "\n")


    # Split train and test data
    if validation:
        train_len = int(len(dataset.dataset)*0.8)
        test_len = len(dataset.dataset) - train_len
        trainset, testset = torch.utils.data.random_split(dataset.dataset, [train_len, test_len])
        # trainset = trainset.to(device)
        trainset = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                num_workers=num_workers, pin_memory=False)
        testset = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                num_workers=num_workers, pin_memory=False)
    else:
        trainset = dataset
        print("false")
    # mean, std = collector.normalization_parameter(testset)
    # print("mean={}, std={}".format(mean, std))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        running_loss = 0.0
        # running_corrects = 0

        # Iterate over data.
        for i, item in enumerate(trainset):
            inputs, labels = item
            # if device is not None:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            if ssout:
                print("({}/{})Batch loss: {}".format(i+1, len(trainset), loss.item()), end="\r")
            # running_corrects += torch.sum(preds == labels.data)

        print()
        # Compute Loss
        epoch_loss = running_loss / len(dataset.dataset)
        print('Loss: {:.4f}'.format(epoch_loss))

        # Evaluate model
        if validation:
            score = test_model(model, testset, device=device, ssout=ssout)
            print('Score: ')
            print('\tf_measure(weighted)={}\n\tf_measure(macro)={}\n\ttop_k={}'.format(score["f_weighted"], score["f_macro"], score["top_k"]))

        decay.step()

        with open(history, 'a') as f:
            row = f"{epoch}; {epoch_loss}"
            if validation:
                row += f"; {score}"    
            f.write(row + "\n")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model, history

#########################################################
###############  MODEL VISUALISATOR  ####################
#########################################################

# def save_history(history, filename):
#     with open(filename, "w+") as f:
#         json.dump(history, f)

def view_history(filename):
    pass

#########################################################
##################  DATA SELECTION  #####################
#########################################################


# def get_sampler(filename, class_to_idx):

    # freq = np.genfromtxt(filename, delimiter=';', dtype='int')
    # counts = freq[:,0]
    # labels = freq[:,1]

    # class_weights = [1/c for c in counts]
    # example_weights = [class_weights[np.where(labels == int(idx2cls[image[1]]))[0][0]] for image in dataset]
    # # print(example_weights)
    # # class to index : {134536:0, 1738392:1, }
    # sampler = torch.utils.data.WeightedRandomSampler(example_weights, len(class_to_idx))
    
    # return sampler
# def get_sampler(filename, class_to_idx):

#     freq = np.genfromtxt(filename, delimiter=';', dtype='int')
#     counts = freq[:,0]
#     labels = freq[:,1]

#     class_weights = [1/c for c in counts]
#     # print(np.where(labels == int(classe))[0][0] for classe in class_to_idx)
#     example_weights = [class_weights[np.where(labels == int(classe))[0][0]] for classe in class_to_idx.keys()]
#     # print(example_weights)

#     # class to index : {134536:0, 1738392:1, }
#     sampler = torch.utils.data.sampler.WeightedRandomSampler(example_weights, num_samples=len(class_to_idx), replacement=True)

#     #Let there be 9 samples and 1 sample in class 0 and 1 respectively
#     # class_counts = [9.0, 1.0]
#     # file = np.genfromtxt(filename, delimiter=";", dtype="int")
#     # freq, labels = file[:, 0], file[:,1]
#     # print(len(freq), len(labels))
#     # num_samples = sum(freq)
#     # # labels = [0, 0,..., 0, 1] #corresponding labels of samples

#     # class_weights = [num_samples/freq[i] for i in range(len(freq))]
#     # print(labels)
#     # weights = [class_weights[labels[class_to_idx[i]]] for i in range(int(sum(counts)))]
#     # print("weights: ", weights)
#     # sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
#     # sampler=None
#     return sampler


