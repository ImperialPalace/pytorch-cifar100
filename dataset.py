""" train and test dataset

author baiyu
"""
import json
import os
import sys
import pickle

import random
# from skimage import io
# import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image

class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

class ImageNet1000Train(Dataset):
    def __init__(self, data_path, transform=None):
        if not isinstance(data_path, list):
            data_path = [data_path]

        if not os.path.exists('./output'):
            os.makedirs('./output')

        names = []
        for path in data_path:
            for sub_class in os.listdir(path):
                if sub_class != 'ahand':
                    if not sub_class in names:
                        names.append(sub_class)

        def save_dict(filename, dic):
            '''save dict into json file'''
            with open(filename, 'w') as json_file:
                json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)

        def save_label(filename, dic):
            with open(filename, 'w') as fd:
                fd.writelines([line + '\n' for line in sorted(dic)])

        self.class_to_idx = dict(zip(sorted(names), range(len(names))))
        save_dict('./output/class_to_idx.json',self.class_to_idx)
        save_label('./output/classes.txt', names)

        print('Save:{},{}'.format('./output/class_to_idx.json', './output/classes.txt'))

        normalize = transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
        self.transform = transforms.Compose([transforms.Resize((252, 252)), transforms.RandomCrop(224),
                                                 transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

        self.images, self.labels = [], []
        for path in data_path:
            for sub_class in os.listdir(path):
                if sub_class != 'ahand':
                    sub_folder = os.path.join(path, sub_class)
                    for sub_file in sorted(os.listdir(sub_folder)):
                        self.images.append(os.path.join(sub_folder, sub_file))
                        self.labels.append(self.class_to_idx[sub_class])

        random.seed(1000)
        random.shuffle(self.images)

        random.seed(1000)
        random.shuffle(self.labels)

        self.images  = self.images[:len(self.images)//2]
        self.labels = self.labels[:len(self.labels) // 2]

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.images)


class ImageNet1000Test(Dataset):

    def __init__(self, data_path, transform=None):
        if not isinstance(data_path, list):
            data_path = [data_path]

        normalize = transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])

        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

        self.images, self.labels = [], []
        for path in data_path:
            for sub_class in os.listdir(path):
                if sub_class != 'ahand':

                    sub_folder = os.path.join(path, sub_class)
                    for sub_file in sorted(os.listdir(sub_folder)):
                        self.images.append(os.path.join(sub_folder, sub_file))
                        self.labels.append(self.class_to_idx[sub_class])

        random.seed(1000)
        random.shuffle(self.images)

        random.seed(1000)
        random.shuffle(self.labels)

        self.images = self.images[len(self.images) // 2:]
        self.labels = self.labels[len(self.labels) // 2:]

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.images)

class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return image, label

class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return image, label

