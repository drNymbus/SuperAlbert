#from sklearn.metrics import f1_score, top_k_accuracy_score

import numpy as np
import math
import time
import os
import json

import collector
from SuperAlbert.cce import CCE
from SuperAlbert.model import *

def train_model(model, dataset, criterion, optimizer, decay, num_epochs=1, device=None, NB_CLASSES=1081):
    since = time.time()
    history = {}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0

        # Split epoch data
        train_len = int(len(dataset.dataset)*0.8)
        test_len = len(dataset.dataset) - train_len
        trainset, testset = torch.utils.data.random_split(dataset.dataset, [train_len, test_len])

        trainset = torch.utils.data.DataLoader(dataset=trainset)
        testset = torch.utils.data.DataLoader(dataset=testset)

        # Iterate over data.
        for inputs, labels in trainset:
            # print(type(inputs), type(labels))
            # print(labels)
            if device is not None:
                inputs = inputs.to(device)
                labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                outputs = model(inputs)
                # print("output", outputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            # running_corrects += torch.sum(preds == labels.data)

        # Compute Loss
        epoch_loss = running_loss / len(dataset.dataset)
        print('{} Loss: {:.4f}'.format(epoch, epoch_loss))

        # Evaluate model
        score = test_model(model, testset, device=device, NB_CLASSES=NB_CLASSES)
        print('{} Score: f_measure(weighted)={}, f_measure(macro)={}, top_k={}'.format(epoch, score["f_weighted"], score["f_macro"], score["top_k"]))

        decay.step()

        history[epoch] = {
            "loss" : epoch_loss,
            "score" : score
        }
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model, history

def test_model(model, dataset, device=None, NB_CLASSES=1081):
    model.eval()

    predictions = []
    y_true = []

    predictions_score = []
    y_true_flat = []

    for inputs, labels in dataset:
        if device is not None:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # labels_flat = torch.FloatTensor([0 if x != labels[0] else 1 for x in range(NB_CLASSES)])

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            # Special case for inception because in training it has an auxiliary output. In train
            #   mode we calculate the loss by summing the final output and the auxiliary output
            #   but in testing we only consider the final output.
            outputs = model(inputs)

            predictions.append(outputs.argmax(dim=1).item())
            predictions_score.append(list(outputs.numpy()[0]))

            y_true.append(labels.item())
            # y_true_flat.append(labels_flat)

    print(y_true, predictions_score)

    #f_score = f1_score(y_true, predictions, average="weighted")
    #f_score_macro = f1_score(y_true, predictions, average="macro")
    # top_k_score = top_k_accuracy_score(y_true, predictions_score)

    return {
        'f_weighted': -1,#f_score,
        'f_macro' : -1,#f_score_macro,
        'top_k': -1
    }

if __name__ == "__main__":
    # data loading
    data_loaders, image_datasets, idx_to_class = collector.get_datasets("/home/data/challenge_2022_miashs/", batch_size=128)
    trainset, testset = data_loaders["train"], data_loaders["test"]
    img_train, img_test = image_datasets["train"], image_datasets["test"]

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init model
    model = create_model(None, len(idx_to_class))
    model = model.to(device)

    # define loss, optimizer and learning rate
    optimizer_ft = optim.SGD(model.classifier[1].parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # criterion = nn.CrossEntropyLoss()
    criterion = CCE(device=device)
    learning_rate_decay = MultiStepLR(optimizer_ft, milestones=[30, 40], gamma=0.1)

    # Train and evaluate
    model, history = train_model(model, trainset, criterion, optimizer_ft, learning_rate_decay, num_epochs=5, device=device, NB_CLASSES=len(idx_to_class))

    torch.save(model.state_dict(), 'model_efficient_cce.torch')
    with open("logfile_efficientCCE.log", 'w') as f:
        json.dump(history, f)
