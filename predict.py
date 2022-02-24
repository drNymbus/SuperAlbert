import os
import collector
from SuperAlbert.model import *
import utils
import pandas as pd

def get_predictions(model, dataset, img_set, idx_to_class, device=None, ssout=False):
    model.eval()

    answers = []
    idx = 0

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
            prediction = outputs.argmax(dim=1)
            for i, a in enumerate(prediction):
                predicted_class = a.cpu().numpy()
                s = img_set.imgs[idx][0]
                answers.append((os.path.basename(os.path.splitext(s)[0]), int(idx_to_class[int(predicted_class)]), list(outputs.cpu().numpy()[i])))
                idx += 1

        if ssout:
            print("({}/{}) Prediction ...".format(i+1, len(dataset)), end="\r")

    print()
    return answers

def save_predictions(answers, filename):
    # save predictions in filename (CSV file format)
    with open(filename, 'w') as f:
        f.write('Id,Category,Confidence\n')
        for k, p, pp in answers:
            f.write('{},{},{}\n'.format(k, p, pp))

def load_prediction_to_submission(filename, filename_result):
    df = pd.read_csv(filename)
    df = df.iloc[:, :2]
    if df.isnull().values.any():
        df = df.dropna()
        df = df.reset_index(drop=True)
    if type(df.iloc[0, 1])!='numpy.int64':
        df['Category'] = df['Category'].astype(int)
    df.to_csv(filename_result, index=False)

def create_submission(answers, filename):
    # save predictions in filename (CSV file format)
    with open(filename, 'w') as f:
        f.write('Id,Category\n')
        for k, p, _ in answers:
            f.write('{},{}\n'.format(k, p))

# def save_prediction(answers, filename):
#     # save predictions in filename (CSV file format)
#     with open(filename, 'w') as f:
#         f.write('Id,Category\n')
#         for k, p, _ in answers:
#             f.write('{},{}\n'.format(k, p))

# def predict(model, dataset, img_set, idx_to_class, device=None, filename="answers_challenge.csv"):
#     model.eval()

#     answers = []
#     idx = 0
#     for inputs, labels in dataset:
#         if device is not None:
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#         # forward
#         # track history if only in train
#         with torch.set_grad_enabled(False):
#             # Get model outputs and calculate loss
#             # Special case for inception because in training it has an auxiliary output. In train
#             #   mode we calculate the loss by summing the final output and the auxiliary output
#             #   but in testing we only consider the final output.
#             outputs = model(inputs)
#             prediction = outputs.argmax(dim=1)
#             for a in prediction:
#                 predicted_class = a.cpu().numpy()
#                 s = img_set.imgs[idx][0]
#                 answers.append((os.path.basename(os.path.splitext(s)[0]), idx_to_class[int(predicted_class)]))
#                 idx += 1

#     # save predictions in filename (CSV file format)
#     with open(filename, 'w') as f:
#         f.write('Id,Category\n')
#         for k, p in answers:
#             f.write('{},{}\n'.format(k, p))

if __name__ == "__main__":
    # print("Toto")
    path = "/home/miashs3/SuperAlbert/results/2022-02-24 19:48:14.426818_b3_CE_1/"
    data_loaders, image_datasets, idx_to_class = collector.get_datasets("/home/data/challenge_2022_miashs", batch_size=128)
    trainset, testset = data_loaders["train"], data_loaders["test"]
    img_train, img_test = image_datasets["train"], image_datasets["test"]

    # Detect if we have a GPU available
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    model = create_model_b3(len(idx_to_class))
    model = utils.load_model(model, path+"model.torch")
    model = model.to(device)
    answers = get_predictions(model, testset, img_test, idx_to_class, device=device)
    create_submission(answers, path+'predictions.csv')
