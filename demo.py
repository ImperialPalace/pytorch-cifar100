# -----------------------------------------------------
# @Time    : 2021/3/31 14:20
# @Author  : 
# @Software: 
# -----------------------------------------------------
# -*- coding: utf-8 -*-

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from binascii import a2b_hex
import os
from models.resnet import resnet18

import argparse
np.set_printoptions(precision=6, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument('-w', help='weight of model', default='')
parser.add_argument('-d', help='data path', default='')
parser.add_argument('-c', help='label txt', default='')

args = parser.parse_args()

def load_model(weight):
    net = resnet18()
    net.load_state_dict(torch.load(weight))
    net.eval()

    return net.cuda()


def load_image(path):
    img = Image.open(path).convert('RGB').resize((224, 224), resample=Image.BILINEAR)
    return img


def load_classes(class_path):
    with open(class_path, 'r') as f:
        lines = f.readlines()
        lines = [line.replace('\n', '') for line in lines if line.strip() != '']
    return lines


def detect(model, image):
    with torch.no_grad():
        feature = model(image.cuda())

    return feature


def get_topk(data):
    y = torch.argmax(torch.softmax(data, dim=1))
    print("label:", y.cpu().numpy())
    return y


if __name__ == '__main__':
    data_path = 'demo_images'
    model_path = 'checkpoint/resnet18/Monday_11_April_2022_18h_49m_38s/resnet18-11-best.pth'
    class_path = 'output/classes.txt'

    classes = load_classes(class_path)
    sorted(classes)
    
    for idx, item in enumerate(classes):
        print("{}, {}".format(item, a2b_hex(item).decode('utf-8')))

    model = load_model(model_path)

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])])

    for image_file in sorted(os.listdir(data_path)):
        print("====================")

        path = os.path.join(data_path, image_file)
        print(path)
        image = load_image(path)
        image = transform(image)
        image = image.unsqueeze(0)
        feature = detect(model, image)

        class_pred = torch.softmax(feature, dim=1)
        print("class predict:", class_pred.cpu().numpy())

        label = get_topk(feature)

        print(a2b_hex(classes[int(label)]).decode('utf-8'))
