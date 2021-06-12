import os
import cv2
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class DataLoader(Dataset):
    def __init__(self, size, ids, datatype):
        self.size = size
        self.img_ids = ids
        if datatype != "test":
            self.datafolder = "/Users/oyo/PycharmProjects/Segnet/data_semantics/training/data"
        else:
            self.datafolder = "/Users/oyo/PycharmProjects/Segnet/data_semantics/testing/image_2"
        self.datatype = datatype
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((self.size, self.size)), transforms.ToTensor()])

    def __getitem__(self, item):
        img_name = self.img_ids[item]
        if self.datatype != "test":
            img_path = os.path.join(self.datafolder + "/image", img_name)
            label_path = os.path.join(self.datafolder + "/semantic", img_name)
        else:
            img_path = os.path.join(self.datafolder, img_name)
            label_path = None
        img = cv2.imread(img_path)
        if label_path is not None:
            label = self.getlabel(label_path)
            img = self.transform(img)
            #img = torch.unsqueeze(img, 0)
            label = torch.from_numpy(label)
            return img/255, label
        else:
            return img/255, None

    def getlabel(self, label_path):
        l = cv2.imread(label_path)
        label = np.zeros((l.shape[0], l.shape[1], 34))
        for i in range(1,35):
            a = np.array(l[:,:,0])
            a = a + np.ones(a.shape)
            a[a != i] = 0
            label[:,:,i-1] = a
        label = cv2.resize(label, (self.size, self.size))
        label[label > 0] = 1
        label = np.transpose(label, (2,0,1))
        return label

    def __len__(self):
        return len(self.img_ids)
