"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import pytorch_lightning as pl
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(out, labels):
    criterion = nn.MSELoss()

    labels = labels.float()  # Covert tensor.int64 to tensor.float32(model's datatype)
    loss = torch.sqrt(criterion(out, labels))
    return loss

    

class HomographyModel(pl.LightningModule):
    def __init__(self):
        super(HomographyModel, self).__init__()
        self.model = Net()

    def forward(self, a):
        return self.model(a)

    def validation_step(self, img_batch, label_batch):
        delta = self.model(img_batch)
        loss = LossFn(delta, label_batch)
        print("Validation loss", loss)
        return {"val_loss": loss}

    
    def validation_epoch_end(outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

    def conv_block(in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self):
    


        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        
        #############################
        # Fill your network initialization of choice here!
        #############################
    
        
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 16 * 128, 1024)
        self.fc2 = nn.Linear(1024, 8)


    def forward(self, x):

            mini_batch_size = x.shape[0]
            dim_x = x.shape[1]
            dim_y = x.shape[2]
            depth = x.shape[3]

            x = x.view(torch.Size([mini_batch_size, depth, dim_x, dim_y]))

            x = self.conv1(x)
            x = self.conv2(x)
            x = self.maxpool(x)
            x = self.conv2(x)
            x = self.conv2(x)
            x = self.maxpool(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.maxpool(x)
            x = self.conv4(x)
            x = self.conv4(x)
            x = self.dropout(x)

            x = x.view(x.size(0), -1)  # Required prior to passing x to fully connected layer.

            x = self.fc1(x)
            x = self.fc2(x)
            return x

        #Regression Network
    
        #   self.homography_net = nn.Sequential(conv_block(2, 64),     #input is a grayscale image
        #                                     conv_block(64, 64, pool = True),
        #                                     conv_block(64, 64),
        #                                     conv_block(64, 64, pool = True),
        #                                     conv_block(64, 128),
        #                                     conv_block(128, 128, pool = True),
        #                                     conv_block(128, 128),
        #                                     conv_block(128, 128),
        #                                     nn.Dropout2d(0.4),
        #                                     nn.Flatten(),
        #                                     nn.Linear(16*16*128, 1024),
        #                                     nn.Dropout(0.4),
        #                                     nn.Linear(1024, 8))



        # self.homography_net = nn.Sequential(nn.Conv2d(channels, 64, kernel_size=3, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.MaxPool2d(2),
        #                                 nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.MaxPool2d(2),
        #                                 nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.MaxPool2d(2),
        #                                 nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.Dropout2d(0.5),
        #                                 nn.Flatten(),
        #                                 nn.Linear(xinput//8*yinput//8*128, 1024),
        #                                 nn.Dropout(0.5),
        #                                 nn.Linear(1024, 8))

    # def forward(self, x):
    #     """
    #     Input:
    #     x is a MiniBatch of the image
    #     Outputs:
    #     out - output of the network
    #     """

    #     out = self.homography_net(x)
    #     return out

# class HomographyNet(nn.Module):
#     def __init__(self):
#         super(HomographyNet, self).__init__()

#         self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
#         self.batchnorm1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
#         self.batchnorm2 = nn.BatchNorm2d(64)
#         self.pool1 = nn.MaxPool2d(2, stride=2)

#         self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
#         self.batchnorm3 = nn.BatchNorm2d(64)
#         self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
#         self.batchnorm4 = nn.BatchNorm2d(64)
#         self.pool2 = nn.MaxPool2d(2, stride=2)

#         self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
#         self.batchnorm5 = nn.BatchNorm2d(128)
#         self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
#         self.batchnorm6 = nn.BatchNorm2d(128)
#         self.pool3 = nn.MaxPool2d(2, stride=2)

#         self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
#         self.batchnorm7 = nn.BatchNorm2d(128)
#         self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
#         self.batchnorm8 = nn.BatchNorm2d(128)

#         self.dropout1 = nn.Dropout2d(0.5)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(16*16*128, 1024)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(1024, 8)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.batchnorm1(x)
#         x = nn.ReLU(x)
#         x = self.conv2(x)
#         x = self.batchnorm2(x)
#         x = nn.ReLU(x)
#         x = self.pool1(x)

#         x = self.conv3(x)
#         x = self.batchnorm3(x)
#         x = nn.ReLU(x)
#         x = self.conv4(x)
#         x = self.batchnorm4(x)
#         x = nn.ReLU(x)
#         x = self.pool2(x)

#         x = self.conv5(x)
#         x = self.batchnorm5(x)
#         x = n



    