"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# Don't generate pyc codes
sys.dont_write_bytecode = True

def loss_fn(out, labels):
    return torch.sqrt(nn.MSELoss()(labels.float(), out))


# def loss_fn(out, labels):
#     criterion = nn.MSELoss()
#     labels = labels.float()  
#     loss = torch.sqrt(criterion(out, labels))
#     return loss


class SupHomographyModel(pl.LightningModule):
    def __init__(self):
        super(SupHomographyModel, self).__init__()
        self.model = Net()

    def forward(self, a):
        return self.model(a)

    def validation_step(self, img_batch, label_batch):
        delta = self.model(img_batch)
        loss = loss_fn(delta, label_batch)
        print("Validation loss", loss)
        return {"val_loss": loss}
        
    def validation_epoch_end(outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8 * 8 * 256, 512)
        self.fc2 = nn.Linear(512, 8)

    def forward(self, x):
        mini_batch_size = x.shape[0]
        dim_x = x.shape[1]
        dim_y = x.shape[2]
        depth = x.shape[3]
        # x = torch.from_numpy(x)
        x = x.view(torch.Size([mini_batch_size, depth, dim_x, dim_y]))

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # Required prior to passing x to fully connected layer.

        x = self.fc1(x)
        x = self.fc2(x)
        return x



