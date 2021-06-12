import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Segnet(nn.Module):

    def __init__(self):
        super(Segnet, self).__init__()
        self.encoder = encoder().apply(weights_init)
        self.decoder = decoder(34).apply(weights_init)

    def forward(self, x):
        x, ib = self.encoder(x)
        x = self.decoder(x, ib)
        return x

class encoder(nn.Module):

    def __init__(self):
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        '''self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)'''

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x, ib1 = F.max_pool2d_with_indices(x, 2, 2)
        ### block1
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x, ib2 = F.max_pool2d_with_indices(x, 2, 2)
        ### block2
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x, ib3 = F.max_pool2d_with_indices(x, 2, 2)
        ### block3
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)
        x, ib4 = F.max_pool2d_with_indices(x, 2, 2)
        ### block4
        '''x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = F.relu(x)
        x, ib5 = F.max_pool2d_with_indices(x, 2, 2)'''
        ib5 = None
        return x, [ib5, ib4, ib3, ib2, ib1]

class decoder(nn.Module):
    def __init__(self, num_classes):
        super(decoder,self).__init__()
        self.num_classes = num_classes
        '''self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)'''
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, self.num_classes, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(34)

    def forward(self, x, ib):
        '''x = F.max_unpool2d(x, ib[0], 2, 2)
        x = self.conv13(x)
        x = self.bn13(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)'''
        ### block5
        x = F.max_unpool2d(x, ib[1], 2, 2)
        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        ### block4
        x = F.max_unpool2d(x, ib[2], 2, 2)
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        ### block3
        x = F.max_unpool2d(x, ib[3], 2, 2)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        ### block2
        x = F.max_unpool2d(x, ib[4], 2, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.softmax(x, dim=1)
        return x

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight.data)
        nn.init.zeros_(m.bias.data)










