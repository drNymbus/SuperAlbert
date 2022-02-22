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
    return model.load_state_dict(torch.load(filename))

#########################################################
##################  TRAINING  ###########################
#########################################################

def test_model(model, dataset, device=None):
    model.eval()

    predictions = []
    predictions_score = []
    y_true = []

    for inputs, labels in dataset:
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

            predictions.append(outputs.argmax(dim=1).item())
            predictions_score.append(list(outputs.cpu().numpy()[0]))

            y_true.append(labels.item())

    f_score_weighted = f1_score(y_true, predictions, average="weighted")
    f_score_macro = f1_score(y_true, predictions, average="macro")
    #top_k_score = top_k_accuracy_score(y_true, predictions_score)

    return {
        'f_weighted': f_score_weighted,
        'f_macro' : f_score_macro,
        'top_k': -1#top_k_score
    }

def train_model(model, dataset, criterion, optimizer, decay, batch_size=128, num_epochs=5, num_workers=16, device=None):
    since = time.time()
    history = {}

    with open("results/log.txt", 'a') as f:
        f.write("e, loss\n")


    # Split train and test data
    train_len = int(len(dataset.dataset)*0.8)
    test_len = len(dataset.dataset) - train_len
    trainset, testset = torch.utils.data.random_split(dataset.dataset, [train_len, test_len])
    trainset = torch.utils.data.DataLoader(dataset=trainset.to(device), batch_size=batch_size,
                                            num_workers=num_workers, pin_memory=False)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        running_loss = 0.0
        # running_corrects = 0

        # Iterate over data.
        for i, item in enumerate(trainset):
            inputs, labels = item
            # if device is not None:
            #     inputs = inputs.to(device)
            #     labels = labels.to(device)

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
            if (i%100) == 1:
                print("({}/{})Batch loss: {}".format(i+1, len(trainset), loss.item()), end="")
            # running_corrects += torch.sum(preds == labels.data)

        # Compute Loss
        epoch_loss = running_loss / len(dataset.dataset)
        print('{} Loss: {:.4f}'.format(epoch, epoch_loss))

        # Evaluate model
        # testset = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
        #                                         num_workers=num_workers, pin_memory=False)
        # score = test_model(model, testset, device=device)
        # print('{} Score: '.format(epoch))
        # print('\tf_measure(weighted)={}\n\tf_measure(macro)={}\n\ttop_k={}'.format(score["f_weighted"], score["f_macro"], score["top_k"]))

        decay.step()

        # history[epoch] = {
        #     "loss" : epoch_loss,
        #     "score" : score
        # }
        with open("results/log.txt", 'a') as f:
            f.write("{epoch}, {epoch_loss}")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model, history

#########################################################
###############  MODEL VISUALISATOR  ####################
#########################################################

def save_history(history, filename):
    with open(filename, "w+") as f:
        json.dump(history, f)

def view_history(filename):
    pass
