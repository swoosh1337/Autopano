#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import Adam
from Network.Network_supervised import SupHomographyModel, loss_fn
from Network.Network_unsupervised import UnSupHomographyModel
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
from random import choice
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from typing import Dict, List


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")


def GenerateBatch(BasePath: str, train_images: List, labels: Dict, MiniBatchSize: int, set: str = "Train"):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    stacked_batches = []
    h_batch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        ImageNum += 1

        random_original_name = choice(train_images) # Randomly select an image
        original_img_path = os.path.join(BasePath, f"{set}/Original", random_original_name) # Get the path to the image
        warped_img_path = os.path.join(BasePath, f"{set}/Warped", random_original_name) # Get the path to the image

        img1 = cv2.imread(original_img_path)
        img2 = cv2.imread(warped_img_path)

        stacked_images = np.concatenate([img1, img2], axis=2).astype(np.float32) # Concatenate the images
        h4pt = labels[random_original_name] # Get the homography matrix

        stacked_batches.append(torch.from_numpy(stacked_images)) # Append the image to the batch
        h_batch.append(torch.tensor(h4pt)) # Append the homography matrix to the batch


    return torch.stack(stacked_batches).to(device), torch.stack(h_batch).to(device) # Return the batch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


lossTrainList = [] # List to store training loss
lossValList = [] # List to store validation loss

def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = SupHomographyModel().to(device)

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = Adam(model.parameters(), lr=0.005)

    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    val_set_img_names = os.listdir(os.path.join(BasePath, "Val/Original/"))
    epochlosslist = []
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            I1Batch, CoordinatesBatch = GenerateBatch(
                BasePath, DirNamesTrain, TrainCoordinates, MiniBatchSize
            )

            
            model.train()
            PredicatedCoordinatesBatch = model(I1Batch)
            LossThisBatch = loss_fn(PredicatedCoordinatesBatch, CoordinatesBatch)
            lossTrainList.append(LossThisBatch)
           

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:

                # Write training losses to tensorboard
                Writer.add_scalar(
                    "LossEveryIter",
                    LossThisBatch,
                    Epochs * NumIterationsPerEpoch + PerEpochCounter,
                )
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()

                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")
        epochlosslist.append(sum(lossTrainList) / len(lossTrainList))
      
        model.eval()
        with torch.no_grad():
            val_batch, val_labels = GenerateBatch(
                BasePath, val_set_img_names, TrainCoordinates, MiniBatchSize, "Val"
            )
            result = model.validation_step(val_batch, val_labels)
            lossValList.append(result["val_loss"])
        # Write validation losses to tensorboard
        Writer.add_scalar(
            "LossEveryEpoch",
            result["val_loss"],
            Epochs,
        )
        # If you don't flush the tensorboard doesn't update until a lot of iterations!
        Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")
    with open("Loss.txt", "w") as f:
        f.write(str(epochlosslist))
    with open("Val_loss.txt", "w") as f:
        f.write(str(lossValList))
   

def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default='../Data/',
        help="Base path of images, Default:../Data/",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=100,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:64",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=99,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:99",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
    )


if __name__ == "__main__":
    main()
