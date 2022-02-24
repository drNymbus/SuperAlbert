import torch
import torch.nn as nn

import timm
from timm.optim import create_optimizer_v2 as create_optimizer

import warnings
import time
import datetime

import utils
import collector
import predict

from SuperAlbert.cce import CCE
from SuperAlbert.model import *

if __name__ == "__main__":
    since = time.time()
    warnings.filterwarnings('ignore', '.*interpolation.*', )
    # chillout timm

    suffix = "b3_CE_1"
    RESULTS_PATH = utils.create_model_dir("{}_{}".format(datetime.datetime.now(), suffix))

    NB_CLASS = 1081

    EPOCHS = 20
    BATCH_SIZE = 1
    NUM_WORKERS = 16

    SSOUT = True

    device = utils.get_device()
    #print(device)

    # Data loading
    # idx2cls, cls2idx = collector.get_indices_and_classes("../data_testing/train")
    # SAMPLER = utils.get_sampler("./data_aux/frequencies.csv", cls2idx)
    # idx2cls, cls2idx = collector.get_indices_and_classes("/home/data/challenge_2022_miashs/train")
    # SAMPLER = utils.get_sampler("/home/miashs3/SuperAlbert/data_aux/frequencies.csv", cls2idx)

    # trainset, train_img = collector.get_dataloader("../data_testing/train", sampler="./data_aux/frequencies.csv", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    trainset, train_img = collector.get_dataloader("/home/data/challenge_2022_miashs/train", sampler="/home/miashs3/SuperAlbert/data_aux/frequencies.csv", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Init model
    model = create_model_b3(NB_CLASS)
    model = model.to(device)

    # Define loss, optimizer and learning rate
    optimizer_ft = create_optimizer(model, 'sgd', learning_rate=0.01, momentum=0.9, weight_decay=1e-4)
    # optimizer_ft = optim.SGD(model.classifier[1].parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # criterion = CCE(device=device)
    criterion = nn.CrossEntropyLoss()
    learning_rate_decay = MultiStepLR(optimizer_ft, milestones=[15, 18], gamma=0.1)

    # Train and evaluate
    history_path = RESULTS_PATH + "history.csv"
    model, history = utils.train_model(model, trainset, False,
                                       criterion, optimizer_ft, learning_rate_decay,
                                       batch_size=BATCH_SIZE, num_epochs=EPOCHS,
                                       num_workers=NUM_WORKERS, device=device, history=history_path,
                                       ssout=SSOUT)

    utils.save_model(model, RESULTS_PATH + "model.torch")
    print("Model saved ...")
    # utils.save_history(history, RESULTS_PATH + "hitory_log.json")


    # Generate Predictions
    # testset, test_img = collector.get_dataloader("../data_testing/", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    testset, test_img = collector.get_dataloader("/home/data/challenge_2022_miashs/test/", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    answers = predict.get_predictions(model, testset, test_img, idx2cls, device=device, ssout=SSOUT)
    print("Predictions done ...")
    predict.save_predictions(answers, RESULTS_PATH + "prediction.csv")

    time_elapsed = time.time() - since
    print("Pipeline terminated after {}m {}s".format(time_elapsed//60, time_elapsed%60))
