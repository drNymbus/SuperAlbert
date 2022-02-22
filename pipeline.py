import utils
import collector
import predict

from SuperAlbert.cce import CCE
from SuperAlbert.model import *

if __name__ == "__main__":
    RESULTS_PATH = utils.create_model_dir("efficientnet_CCE_v1")
    BATCH_SIZE = 256
    # data loading
    data_loaders, image_datasets, idx_to_class = collector.get_datasets("../data/", batch_size=BATCH_SIZE, num_workers=4)
    # data_loaders, image_datasets, idx_to_class = collector.get_datasets("/home/data/challenge_2022_miashs/", batch_size=BATCH_SIZE, num_workers=16)
    trainset, testset = data_loaders["train"], data_loaders["test"]
    img_train, img_test = image_datasets["train"], image_datasets["test"]

    device = utils.get_device()
    #print(device)

    # init model
    model = create_model(len(idx_to_class))
    model = model.to(device)

    # define loss, optimizer and learning rate
    optimizer_ft = optim.SGD(model.classifier[1].parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = CCE(device=device)
    learning_rate_decay = MultiStepLR(optimizer_ft, milestones=[7, 10], gamma=0.1)

    # Train and evaluate
    #print(device)
    model, history = utils.train_model(model, trainset, criterion, optimizer_ft, learning_rate_decay, batch_size=BATCH_SIZE, num_epochs=12, device=device)

    utils.save_model(model, RESULTS_PATH + "model.torch")
    utils.save_history(history, RESULTS_PATH + "hitory_log.json")

    # Generate Predictions
    answers = predict.get_predictions(model, testset, img_test, idx_to_class, device=device)
    predict.save_predictions(answers, RESULTS_PATH + "prediction.csv")
