import time

import utils
import collector
import predict

from SuperAlbert.cce import CCE
from SuperAlbert.model import *

if __name__ == "__main__":
    since = time.time()
    RESULTS_PATH = utils.create_model_dir("efficientnet_b4_CCE_v2")

    EPOCHS = 1
    BATCH_SIZE = 128
    NUM_WORKERS = 16

    device = utils.get_device()
    #print(device)

    # Data loading
    # data_loaders, image_datasets, idx_to_class = collector.get_datasets("../data_testing/", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, device=device)
    data_loaders, image_datasets, idx_to_class = collector.get_datasets("/home/data/challenge_2022_miashs/",
                                                                        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, device=device)
    trainset, testset = data_loaders["train"], data_loaders["test"]
    img_train, img_test = image_datasets["train"], image_datasets["test"]

    # Init model
    model = create_model_b3(len(idx_to_class))
    model = model.to(device)

    # Define loss, optimizer and learning rate
    optimizer_ft = optim.SGD(model.classifier[1].parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = CCE(device=device)
    learning_rate_decay = MultiStepLR(optimizer_ft, milestones=[10, 12], gamma=0.1)

    # Train and evaluate
    history_path = RESULTS_PATH + "history.csv"
    model, history = utils.train_model(model, trainset, criterion, optimizer_ft, learning_rate_decay, batch_size=BATCH_SIZE, num_epochs=EPOCHS, num_workers=NUM_WORKERS, device=device, history=history_path, ssout=True)

    utils.save_model(model, RESULTS_PATH + "model.torch")
    print("Model saved ...")
    # utils.save_history(history, RESULTS_PATH + "hitory_log.json")


    # Generate Predictions
    answers = predict.get_predictions(model, testset, img_test, idx_to_class, device=device, ssout=True)
    print("Predictions done ...")
    predict.save_predictions(answers, RESULTS_PATH + "prediction.csv")

    time_elapsed = time.time() - since
    print("Pipeline terminated after {}m {}s".format(time_elapsed//60, time_elapsed%60))