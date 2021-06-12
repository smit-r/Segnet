import os
import pickle
import cv2
import inline as inline
import matplotlib
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch import optim

import matplotlib.pyplot as plt

from model import encoder, Segnet
from dataLoader import DataLoader


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def entropy(labels, output):
    l = -torch.sum(torch.mul(labels, torch.log(output)))/(output.shape[1]*output.shape[2])
    return l

def plot_grad_flow(named_parameters, step):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    if not (os.path.exists("/Users/oyo/PycharmProjects/Segnet/gradflow.p")):
        fig = plt.figure(100)
    else:
        fig = pickle.load(open("/Users/oyo/PycharmProjects/Segnet/gradflow.p", "rb"))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    pickle.dump(fig, open("/Users/oyo/PycharmProjects/Segnet/gradflow.p", "wb"))
    plt.show()

def maskToRGB(mask):
    dim = mask.shape
    img = np.zeros((dim[1],dim[2], 3))
    rgb = [0,0,0]
    for i in range(dim[0]):
        rgb[i%3] += 20
        indices = np.where(mask[i] == 1)
        indices = (indices[0], indices[1])
        img[indices] = rgb
    plt.figure(200)
    plt.imshow(img/255)
    plt.show()
    return img

def scoresToMask(scores):
    m = torch.amax(scores, 0)
    m = m.expand(34,-1,-1)
    mask = scores - m
    mask[mask == 0] = 1
    mask[mask < 0] = 0
    return mask


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_dir = "/Users/oyo/PycharmProjects/Segnet/data_semantics/training/data/image"
    dataset = DataLoader(224, os.listdir(data_dir), "train")
    dataLoaderNorm = torch.utils.data.DataLoader(dataset, len(os.listdir(data_dir)), True)
    data, _ = next(iter(dataLoaderNorm))
    mean = torch.mean(data, 0)
    std = torch.std(data, 0)
    mean = mean.expand(5, -1, -1, -1)
    std = std.expand(5, -1, -1, -1)
    print(mean.shape, std.shape, data.shape)
    dataLoader = torch.utils.data.DataLoader(dataset, 5, True)
    epochs = 100
    model = Segnet()
    #model.load_state_dict(torch.load("/Users/oyo/PycharmProjects/Segnet/checkpoints/Segnet_5.pth"))
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    dataLoss = []
    for i in range(epochs):
        steps = 0
        for data, label in dataLoader:
            data = (data - mean)/std
            out = model(data)
            optimizer.zero_grad()
            l = entropy(label, out)
            l.backward()
            #plot_grad_flow(model.named_parameters(), steps) if steps%10 == 0 else plt.show()
            optimizer.step()
            #maskToRGB(label[0])
            print("epochs : %d steps : %d entropyLoss : %d" % (i, steps, l))
            dataLoss.append(l.detach().numpy())
            steps += 1
        testpath = "/Users/oyo/PycharmProjects/Segnet/data_semantics/testing/image_2/000000_10.png"
        testimg = cv2.imread(testpath)
        transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor()])
        testimg = torch.unsqueeze(transform(testimg), 0)
        testlabel = model(testimg)
        mask = scoresToMask(testlabel[0])
        maskToRGB(mask)
        path = "/Users/oyo/PycharmProjects/Segnet/checkpoints/" + "Segnet_" + str(i) + ".pth"
        torch.save(model.state_dict(), path)
        plt.figure(300)
        plt.plot(np.arange(len(dataLoss)), dataLoss)
        plt.show()

