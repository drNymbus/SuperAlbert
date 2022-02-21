import time
import os

import collector
from SuperAlbert.resnet_model import *

def train_model(model, dataset, criterion, optimizer, decay, num_epochs=1, device=None):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataset:
            if device is not None:
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
            # running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(dataset.dataset)

        print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        decay.step()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model

def test_model(model, dataset):
    pass

def predict(model, X):
    pass

if __name__ == "__main__":
    # data loading
    trainset, testset, idx_to_class = collector.get_dataset("/home/data/challenge_2022_miashs/")

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(len(idx_to_class))
    model = model.to(device)
    print(model)

    optimizer_ft = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    learning_rate_decay = MultiStepLR(optimizer_ft, milestones=[30, 40], gamma=0.1)

    # Train and evaluate
    model = train_model(model, trainset, criterion, optimizer_ft, learning_rate_decay, num_epochs=1, device=device)

    # torch.save(model.state_dict(), 'model.torch')

    # test_model(model, trainset, )
    # print(trainset, testset)
