import numpy as np
import cv2
import argparse
import os

from torchvision import models, transforms
from torch.nn import functional as F
from torch import topk

from SuperAlbert.model import *
import collector

# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def show_cam(CAMs, width, height, orig_image, class_idx, all_classes, save_name, folder_path):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + orig_image * 0.5
        # put class label text on the result
        cv2.putText(result, all_classes[class_idx[i]], (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        # cv2.imshow('CAM', result/255.)
        # cv2.waitKey(0)
        cv2.imwrite(f"CAM_outputs/CAM_{folder_path}_{save_name}.jpg", result)


def load_synset_classes(file_path):
    # load the synset text file for labels
    all_classes = []
    with open(file_path, 'r') as f:
        all_lines = f.readlines()
        labels = [line.split('\n') for line in all_lines]
        for label_list in labels:
            current_class = [name.split(',') for name in label_list][0][0][10:]
            all_classes.append(current_class)
    return all_classes


def do_cam(image_path):

    # model_type = 'efficientnet' # 'resnet'

    # read and visualize the image
    image = cv2.imread(image_path)
    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    data_loaders, image_datasets, idx_to_class = collector.get_datasets("/home/data/challenge_2022_miashs", batch_size=128, num_workers=16)

    # get all the classes in a list
    all_classes = list(idx_to_class.values())
    # all_classes = ['tench', 'goldfish', 'great white shark', 'tiger shark', ... ]
    
    # load the model
    # model = models.resnet18(pretrained=True).eval()
    #if model_type == 'resnet':
    #    model = create_pretrained_resnet(len(all_classes)).eval()

    # EfficientNet
    #if model_type == 'efficientnet':
    model = create_model_b3(len(idx_to_class)).eval()
    model.load_state_dict(torch.load('results/efficientnet_b3_CCE_v1/model.torch'))
    
    # hook the feature extractor
    # https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    
    #print(model)
    #Resnet 
    #if model_type == 'resnet':
    #    model._modules.get("layer4").register_forward_hook(hook_feature)
    
    #EfficientNet 
    #if model_type == 'efficientnet':
    model._modules.get("features").register_forward_hook(hook_feature)
    
    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    # define the transformer, resize => tensor => normalize
    transformer = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        ])

    # apply the image transformer
    image_tensor = transformer(image)
    # add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    # forward pass through model
    outputs = model(image_tensor)
    
    # get the softmax probabilities
    probs = F.softmax(input=outputs, dim=outputs.shape[0]).data.squeeze()
    # get the class indices of top k probabilities
    class_idx = topk(probs, 1)[1].int()

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
    # file name to save the resulting CAM image with
    save_name = f"{image_path.split('/')[-1].split('.')[0]}"
    folder_path = image_path.split('/')[-2]
    # show and save the results
    show_cam(CAMs, width, height, orig_image, class_idx, all_classes, save_name, folder_path)


if __name__ == "__main__":
 
    directory = '/home/data/challenge_2022_miashs/train/1365961'
    # folder with 20 images : '1698065', '1365961'

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            do_cam(image_path=directory + '/' + filename)
            continue
        else:
            continue

    